#!/usr/bin/env python3
"""Owned-service entry: numeric request/iteration records, unchanged kernels.

Does not modify installed files, request messages, model computation or routing.
The driver must record this instrumentation and measure profiling separately.
"""
import dataclasses
import functools
import inspect
import json
import os
import time

# The source deployment enables a Rust frontend, which bypasses this Python
# metrics logger. All controlled E1/E2/E3 services use the same Python frontend;
# this is a recorded experimental setting, not a change to shared deployments.
os.environ['VLLM_USE_RUST_FRONTEND'] = '0'
os.environ.pop('VLLM_RUST_FRONTEND_PATH', None)

from vllm.v1.metrics.loggers import PrometheusStatLogger

FIELDS = ('request_id', 'e2e_latency', 'num_prompt_tokens', 'num_preemptions',
          'num_generation_tokens', 'queued_time', 'prefill_time', 'inference_time',
          'decode_time', 'mean_time_per_output_token', 'num_cached_tokens')
original = PrometheusStatLogger.record
signature = inspect.signature(original)
if 'iteration_stats' not in signature.parameters:
    raise RuntimeError('Unverified metrics logger signature')


def emit(kind, row):
    try:
        os.write(1, (kind + ' ' + json.dumps(row, ensure_ascii=False) + '\n').encode())
    except (OSError, TypeError) as exc:
        os.write(2, ('DECODE_METRICS_ERROR ' + type(exc).__name__ + '\n').encode())


@functools.wraps(original)
def record(self, *args, **kwargs):
    bound = signature.bind(self, *args, **kwargs)
    bound.apply_defaults()
    now = time.time()
    iteration = bound.arguments.get('iteration_stats')
    if iteration is not None:
        for request in iteration.finished_requests:
            emit('DECODE_REQUEST_METRICS', {**{key: getattr(request, key, None) for key in FIELDS},
                 'logged_unix': now, 'engine_idx': bound.arguments.get('engine_idx')})
    scheduler = bound.arguments.get('scheduler_stats')
    if scheduler is not None:
        row = {key: getattr(scheduler, key, None) for key in
               ('num_running_reqs', 'num_waiting_reqs', 'step_counter', 'kv_cache_usage')}
        details = getattr(scheduler, 'iteration_details', None)
        row['iteration_details'] = dataclasses.asdict(details) if details is not None and dataclasses.is_dataclass(details) else None
        evictions = getattr(scheduler, 'kv_cache_eviction_events', [])
        row['evictions'] = [dataclasses.asdict(x) for x in evictions if dataclasses.is_dataclass(x)]
        row['logged_unix'] = now
        emit('DECODE_ITERATION_METRICS', row)
    return original(self, *args, **kwargs)


PrometheusStatLogger.record = record

if __name__ == '__main__':
    emit('DECODE_INSTRUMENTATION_READY', {'record_signature': str(signature), 'fields': FIELDS,
                                        'frontend':'Python; numeric logger enabled'})
    from vllm.entrypoints.cli.main import main
    main()
