#!/usr/bin/env python3
"""Equal-budget fixed-P, fixed-D and calibrated PPD-style routing client.

Runs against already provisioned, owned B200 services. It neither allocates
extra hardware nor uses future measured latency/output to choose a route.
The connector protocol (one-way by default) must be held constant across modes.
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from decode_probe_client import encode_body, request_json, stream_request


def common_prefix_length(previous, current):
    shared = 0
    for a, b in zip(previous,current):
        if a != b:
            break
        shared += 1
    return shared


def score(entry, ttft_weight=1, tpot_weight=1):
    if entry['pd_ttft_ms'] <= 0 or entry['pd_tpot_ms'] <= 0:
        raise ValueError('Invalid calibration denominators')
    return (ttft_weight * (entry['pd_ttft_ms'] - entry['local_ttft_ms']) / entry['pd_ttft_ms']
            - tpot_weight * (entry['local_tpot_ms'] - entry['pd_tpot_ms']) / entry['pd_tpot_ms'])


def route(config, turn_index, history, append, output_budget):
    if turn_index == 0:
        return 'pd', None
    mode = config['mode']
    if mode in ('pd', 'local'):
        return mode, None
    if mode != 'dynamic':
        raise ValueError('Unknown policy')
    table = config['lookup_table']
    if not table or any(e.get('split') != 'calibration' for e in table):
        raise ValueError('Dynamic routing requires a separately identified calibration table')
    def distance(entry):
        return (abs(math.log2((history + 1) / (entry['history_length'] + 1)))
                + abs(math.log2((append / output_budget) / (entry['append_length'] / entry['output_budget'])))
                + abs(math.log2(config['request_rate'] / entry['request_rate'])))
    selected = min(table, key=distance)
    value = score(selected, config.get('ttft_weight', 1), config.get('tpot_weight', 1))
    return ('local' if value > 0 else 'pd'), {'calibration_cell': selected['cell'], 'score': value}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise RuntimeError('Refuse overwrite of PD trial')
    config = json.loads(args.config.read_text())
    body = args.manifest.read_bytes()
    manifest = json.loads(body)
    if config.get('gpu_budget') != 8 or config.get('p_gpus') != 4 or config.get('d_gpus') != 4:
        raise ValueError('Main routing experiment requires fixed P4+D4 budget')
    if config['split'] == 'evaluation' and config['mode'] == 'dynamic':
        calibration_hashes = {e['manifest_sha256'] for e in config['lookup_table']}
        if hashlib.sha256(body).hexdigest() in calibration_hashes:
            raise ValueError('Do not evaluate on the calibration manifest')
    rows = []
    lock = threading.Lock()
    start = time.monotonic()
    started_unix = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output.with_suffix('.requests.jsonl')
    with raw_path.open('x') as stream:
        def session(entry):
            scheduled = start + entry['arrival_seconds']
            time.sleep(max(0, scheduled - time.monotonic()))
            actual_arrival = time.monotonic() - start
            previous_tokens = []
            for turn_index, request in enumerate(entry['turns']):
                if turn_index:
                    time.sleep(request.get('gap_seconds', 0))
                tokens = request['prompt_token_ids']
                shared = common_prefix_length(previous_tokens,tokens)
                policy, decision = route(config, turn_index, shared,
                                         max(1, len(tokens) - shared), request['max_tokens'])
                ident = f'{config["id"]}-{entry["id"]}-t{turn_index}'
                begun = time.perf_counter()
                prefill_ms = 0
                prefill_usage = None
                try:
                    extra = None
                    if policy == 'pd':
                        payload = encode_body(config['model'], tokens, 1, config['id'], False)
                        payload['kv_transfer_params'] = {'do_remote_decode': True}
                        prefill_started = time.perf_counter()
                        pref = request_json(config['p_url'] + '/v1/completions', payload,
                                            headers={'X-Request-ID': ident + '-P'})
                        prefill_ms = (time.perf_counter() - prefill_started) * 1000
                        prefill_usage = pref.get('usage')
                        transfer = pref.get('kv_transfer_params')
                        if not transfer or not transfer.get('do_remote_prefill'):
                            raise RuntimeError('P did not export a valid transfer descriptor')
                        extra = {'kv_transfer_params': transfer}
                    row = stream_request(config['d_url'], config['model'], tokens,
                                         request['max_tokens'], config['id'], ident, extra=extra)
                    row['d_request_ttft_ms'] = row['ttft_ms']
                    row['d_request_latency_ms'] = row['latency_ms']
                    if row['ttft_ms'] is not None:
                        row['ttft_ms'] += prefill_ms
                    row['latency_ms'] = (time.perf_counter() - begun) * 1000
                except Exception as exc:
                    row = dict(request_id=ident, error=type(exc).__name__ + ': ' + str(exc),
                               ttft_ms=None, tpot_ms=None, latency_ms=(time.perf_counter()-begun)*1000,
                               output_tokens=0, cached_tokens=None, prompt_tokens=None,
                               output_token_ids=[], single_token_itl_ms=[])
                row.update(session=entry['id'], turn=turn_index, route=policy, route_decision=decision,
                           history_length=shared, append_length=len(tokens)-shared,
                           output_budget=request['max_tokens'], prefill_stage_ms=prefill_ms,
                           prefill_usage=prefill_usage, planned_session_arrival_s=entry['arrival_seconds'],
                           actual_session_arrival_s=actual_arrival)
                with lock:
                    rows.append(row)
                    stream.write(json.dumps(row) + '\n')
                    stream.flush()
                if row['error']:
                    break
                previous_tokens = tokens
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(manifest['sessions']), 256)) as pool:
            list(pool.map(session, manifest['sessions']))
    planned = sum(len(s['turns']) for s in manifest['sessions'])
    result = {'status': 'complete', 'platform': 'B200', 'scope': 'fixed-input two/three-turn routing replay, not ASIC results',
              'config': config, 'manifest_sha256': hashlib.sha256(body).hexdigest(),
              'started_unix': started_unix, 'elapsed_seconds': time.monotonic()-start,
              'planned_requests': planned, 'issued_requests': len(rows),
              'success_requests': sum(not r['error'] for r in rows),
              'dependency_unissued_requests': planned-len(rows), 'rows': rows}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','config')}), flush=True)


if __name__ == '__main__':
    main()
