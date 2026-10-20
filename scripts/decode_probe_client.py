#!/usr/bin/env python3
"""Controlled cached-decode and state-pressure probes for an owned service.

Synthetic shape probes are explicitly distinct from AgentX trace replay.
Per-token events use selected-token logprobs (no full prompt-token echo).
Request/iteration server records must be reconciled separately by the runner.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import random
import threading
import time
import urllib.request


def request_json(url, payload, timeout=1800, headers=None):
    body = json.dumps(payload, separators=(',', ':')).encode()
    req = urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json', **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
        return json.loads(raw) if raw.strip() else {'http_status': response.status, 'empty_response': True}


def prompt(length, seed):
    rng = random.Random(seed)
    return [rng.randrange(100, 100000) for _ in range(length)]


def reset_prefix_cache(url, reset_external=False, wait_seconds=60):
    deadline = time.monotonic()+wait_seconds
    while True:
        reply = request_json(url+'/reset_prefix_cache'+('?reset_external=true' if reset_external else ''),{})
        if reply is True or isinstance(reply,dict) and reply.get('success') is True:
            return reply
        if not (reply is False or isinstance(reply,dict) and reply.get('success') is False):
            raise RuntimeError('Cache reset returned an unverified success schema')
        if time.monotonic() >= deadline:
            raise RuntimeError('Cache reset refused while blocks remain held; do not reuse prior KV')
        time.sleep(.5)


def encode_body(model, tokens, output, salt, stream):
    return dict(model=model, prompt=tokens, max_tokens=output, ignore_eos=True,
                temperature=0, seed=42, stream=stream, cache_salt=salt,
                logprobs=0, return_tokens_as_token_ids=True,
                **({'stream_options': {'include_usage': True}} if stream else {}))


def stream_request(url, model, tokens, output, salt, ident, barrier=None, stop=None, extra=None):
    payload = encode_body(model, tokens, output, salt, True)
    if extra:
        payload.update(extra)
    serialization_started = time.perf_counter()
    wire = json.dumps(payload, separators=(',', ':')).encode()
    serialization_ms = (time.perf_counter() - serialization_started) * 1000
    if barrier is not None:
        barrier.wait(timeout=120)
    row = dict(request_id=ident, started_unix=time.time(), expected_output_tokens=output,
               expected_input_tokens=len(tokens), input_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
               serialization_ms=serialization_ms, stream_events=[], error=None, intentional_abort=False)
    begin = time.perf_counter()
    first = None
    usage = {}
    token_ids = []
    req = urllib.request.Request(url + '/v1/completions', data=wire,
                                headers={'Content-Type': 'application/json', 'X-Request-ID': ident})
    try:
        with urllib.request.urlopen(req, timeout=1800) as response:
            row['http_status'] = response.status
            for line in response:
                if stop is not None and stop.is_set():
                    row['intentional_abort'] = True
                    break
                if not line.startswith(b'data:'):
                    continue
                data = line[5:].strip()
                if not data or data == b'[DONE]':
                    continue
                event = json.loads(data)
                if event.get('error'):
                    raise RuntimeError(str(event['error']))
                if event.get('usage'):
                    usage.update(event['usage'])
                for choice in event.get('choices', []):
                    encoded = (choice.get('logprobs') or {}).get('tokens') or []
                    if not encoded:
                        continue
                    arrived = time.perf_counter()
                    if first is None:
                        first = arrived
                    chunk_ids = [int(x.split(':', 1)[1]) for x in encoded]
                    token_ids.extend(chunk_ids)
                    row['stream_events'].append({'offset_s': arrived - begin, 'tokens': len(chunk_ids)})
    except Exception as exc:
        row['error'] = type(exc).__name__ + ': ' + str(exc)
    end = time.perf_counter()
    row.update(usage=usage, output_token_ids=token_ids, finished_unix=time.time(),
               latency_ms=(end - begin) * 1000,
               ttft_ms=(first - begin) * 1000 if first is not None else None)
    row['output_tokens'] = usage.get('completion_tokens', len(token_ids))
    row['cached_tokens'] = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
    row['prompt_tokens'] = usage.get('prompt_tokens')
    row['tpot_ms'] = ((end - first) * 1000 / (row['output_tokens'] - 1)
                      if first is not None and row['output_tokens'] > 1 else None)
    events = row['stream_events']
    row['single_token_itl_ms'] = [(b['offset_s'] - a['offset_s']) * 1000 for a, b in zip(events, events[1:])
                                   if a['tokens'] == b['tokens'] == 1]
    row['multi_token_chunk_count'] = sum(x['tokens'] != 1 for x in events)
    if not row['intentional_abort'] and not row['error']:
        if row['output_tokens'] != output or len(token_ids) != output or first is None:
            row['error'] = 'Output/stream token accounting mismatch'
    return row


def parallel(url, model, prompts, output, salt, stage, stop=None, barrier=True):
    gate = threading.Barrier(len(prompts)) if barrier else None
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futures = [pool.submit(stream_request, url, model, tokens, output, salt,
                               f'{salt}-{stage}-{i}', gate, stop) for i, tokens in enumerate(prompts)]
        return [f.result() for f in futures]


def ensure_success(rows):
    errors = [r for r in rows if r['error']]
    if errors:
        raise RuntimeError(str([(r['request_id'], r['error']) for r in errors]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Refuse overwrite of a trial')
    config = json.loads(args.config.read_text())
    base, model = config['url'], config.get('model', 'decode-study-v41')
    salt = config['id']
    result = {'status': 'running', 'platform': 'B200', 'kind': config['kind'], 'config': config,
              'scope': 'synthetic fixed-shape service probe; not AgentX or real-task results',
              'started_unix': time.time(), 'rows': []}
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temp = args.output.with_suffix('.tmp')
        temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        temp.replace(args.output)
    save()
    try:
        reset = reset_prefix_cache(base,reset_external=bool(config.get('offload_gib_total')))
        result['prefix_reset_response'] = reset
        length, batch = config['history_length'], config['batch']
        append = config.get('append_length', 128)
        seed = config.get('seed', 20261020)
        active = [prompt(length, seed + i) for i in range(batch)]
        if config['kind'] == 'e1':
            warm = parallel(base, model, active, 1, salt, 'history_prime')
            result['rows'].extend(dict(r, stage='history_prime') for r in warm)
            ensure_success(warm)
            if config.get('profile'):
                result['start_profile'] = request_json(base + '/start_profile', {})
            timed = parallel(base, model, [t + prompt(append, seed + 10000 + i) for i, t in enumerate(active)],
                             config.get('output_tokens', 128), salt, 'timed')
            result['rows'].extend(dict(r, stage='timed') for r in timed)
            if config.get('profile'):
                result['stop_profile'] = request_json(base + '/stop_profile', {})
            ensure_success(timed)
        elif config['kind'] == 'e2':
            paused = config['paused_sessions']
            target = prompt(length, seed + 100000)
            warm = parallel(base, model, [target], 1, salt, 'probe_prime')
            result['rows'].extend(dict(r, stage='probe_prime') for r in warm)
            ensure_success(warm)
            for offset in range(0, paused, 4):
                samples = [prompt(length, seed + 200000 + i) for i in range(offset, min(paused, offset + 4))]
                warm = parallel(base, model, samples, 1, salt, f'paused_prime_{offset}')
                result['rows'].extend(dict(r, stage='paused_prime') for r in warm)
                ensure_success(warm)
            warm = parallel(base, model, active, 1, salt, 'active_prime')
            result['rows'].extend(dict(r, stage='active_prime') for r in warm)
            ensure_success(warm)
            stop = threading.Event()
            gate = threading.Barrier(batch)
            with concurrent.futures.ThreadPoolExecutor(max_workers=batch) as pool:
                futures = [pool.submit(stream_request, base, model,
                           t + prompt(append, seed + 10000 + i), config.get('background_output_budget', 8192),
                           salt, f'{salt}-background-{i}', gate, stop) for i, t in enumerate(active)]
                try:
                    time.sleep(config.get('pause_seconds', 2))
                    resumed = stream_request(base, model, target + prompt(append, seed + 300000),
                                             config.get('output_tokens', 128), salt, salt + '-resumed')
                    result['rows'].append(dict(resumed, stage='resumed'))
                    ensure_success([resumed])
                    time.sleep(1)
                finally:
                    stop.set()
                background = [f.result() for f in futures]
                result['rows'].extend(dict(r, stage='background_intentionally_bounded') for r in background)
                ensure_success(background)
        else:
            raise ValueError('Unknown experiment kind')
        result.update(status='complete', finished_unix=time.time())
        save()
        print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'config')}), flush=True)
    except BaseException as exc:
        result.update(status='failed', finished_unix=time.time(), error=str(exc))
        save()
        raise


if __name__ == '__main__':
    main()
