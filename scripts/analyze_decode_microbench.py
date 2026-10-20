#!/usr/bin/env python3
"""Export allowlisted B200 kernel data and counter summaries, without topology."""
import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics


def stat(values):
    return dict(mean=statistics.fmean(values), min=min(values), max=max(values), runs=len(values))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--benchmark-script', type=Path, required=True)
    args = p.parse_args()
    groups = defaultdict(list)
    cases, runs, fingerprints, versions = [], [], {}, set()
    for source in sorted(args.input.glob('micro_v41_*_r[123].json')):
        raw = source.read_bytes()
        data = json.loads(raw)
        if data['status'] != 'complete' or len(data['cases']) != 12 or data['platform'] != 'B200':
            raise ValueError('Incomplete/incorrect-platform run')
        params = data['parameters']
        versions.add((data['torch'],data['vllm']))
        sha = hashlib.sha256(raw).hexdigest()
        fingerprints[source.name] = sha
        runs.append(dict(format=params['extra_format'], cacheMode=params['cache_mode'],
                         repeat=params['repeat'], cases=len(data['cases']), sha256=sha))
        for row in data['cases']:
            key = (row['extra_format'], row['cache_mode'], row['history_length'], row['batch'])
            groups[key].append(row['latency_ms_mean'])
            cases.append({key: row[key] for key in (
                'history_length','batch','query_heads_per_rank','kernel_query_heads','head_padding_ratio',
                'selected_compressed_tokens','swa_tokens','extra_format','extra_record_bytes','allocated_kv_bytes',
                'logical_selected_kv_bytes_not_measured_dram','latency_ms_mean','latency_ms_p50',
                'latency_ms_p90','latency_ms_min','latency_ms_max','numeric_max_abs_error','cache_mode','cuda_graph')}
                | {'repeat': params['repeat']})
    if len(runs) != 9 or len(cases) != 108 or any(len(v) != 3 for v in groups.values()):
        raise ValueError('Expected 9 independent process runs, 108 shapes, 3 repetitions per configuration')
    counters = []
    cgroups = defaultdict(lambda: defaultdict(list))
    for source in sorted((args.input / 'ncu').glob('ncu_v41_*_r[123].csv')):
        raw = source.read_bytes()
        fingerprints['ncu/' + source.name] = hashlib.sha256(raw).hexdigest()
        text = raw.decode()
        position = text.index('"ID","Process ID"')
        rows = list(csv.DictReader(io.StringIO(text[position:])))
        m = re.fullmatch(r'ncu_(v41_nvfp4|v41_fp8)_b(\d+)_r([123])\.csv', source.name)
        if not m or any('Metric Name' not in row or 'Metric Value' not in row for row in rows):
            raise ValueError('Invalid counter file')
        fmt, batch, repeat = m.group(1), int(m.group(2)), int(m.group(3))
        metrics = {row['Metric Name']: float(row['Metric Value'].replace(',', '')) for row in rows}
        units = {row['Metric Name']:row['Metric Unit'] for row in rows}
        if len(metrics) != 8 or metrics['gpu__time_duration.sum'] <= 0 or not all(map(math.isfinite,metrics.values())):
            raise ValueError('Incomplete hardware-counter set')
        if units['dram__bytes_read.sum'] != 'byte' or units['dram__bytes_write.sum'] != 'byte':
            raise ValueError('DRAM metrics must be explicitly in bytes')
        counters.append(dict(format=fmt, batch=batch, repeat=repeat, history_length=65536,
                             kernel=rows[0]['Kernel Name'].split('<',1)[0].strip(),
                             metrics=metrics, metricUnits=units, profiler='Nsight Compute 2026.1.1',
                             cache_control='all; cold-counter replay, separate from latency runs'))
        for name, value in metrics.items():
            cgroups[(fmt,batch)][name].append(value)
    if len(counters) != 24 or any(len(v) != 3 for metrics in cgroups.values() for v in metrics.values()):
        raise ValueError('Expected 24 counter samples with 3 repetitions')
    if len(versions) != 1:
        raise ValueError('Mixed runtime versions')
    torch_version, vllm_version = next(iter(versions))
    result = dict(schemaVersion=2, platform='NVIDIA B200', torchVersion=torch_version,vllmVersion=vllm_version,
                  scope='Single-GPU FlashMLA sparse-attention operator microbenchmark; not Mega, full-model, AgentX or ASIC results',
                  benchmarkScriptSha256=hashlib.sha256(args.benchmark_script.read_bytes()).hexdigest(),
                  cacheFlushMiB=512, blockTokens=128, effectiveQueryHeads=16, kernelQueryHeads=64,
                  selectedCompressedTokens=512, swaTokens=128,
                  latencyDefinition='CUDA-event full FlashMLA API call with CUDA graph; flush excluded; means/ranges across 3 fresh processes',
                  counterDefinition='One main sparse CUDA kernel; actual DRAM bytes, not logical cache footprint; do not mix profiled timing with normal latency',
                  boundary='Head padding is a kernel constraint; its ratio is not an equivalent latency loss or a demonstrated ASIC speedup',
                  runs=runs, cases=cases,
                  summaries=[dict(format=k[0],cacheMode=k[1],historyLength=k[2],batch=k[3],latencyMs=stat(v)) for k,v in sorted(groups.items())],
                  counterSamples=counters,
                  counterSummaries=[dict(format=k[0],batch=k[1],metrics={n:stat(v) for n,v in metrics.items()}) for k,metrics in sorted(cgroups.items())],
                  evidenceSha256=fingerprints)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2,allow_nan=False) + '\n')
    print(json.dumps({'runs':len(runs),'cases':len(cases),'counterSamples':len(counters),'outputBytes':args.output.stat().st_size}))


if __name__ == '__main__':
    main()
