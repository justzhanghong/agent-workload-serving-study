#!/usr/bin/env python3
"""FlashMLA kernel microbenchmark, not full-model or ASIC performance.

Run inside the audited vLLM image with one exclusively available B200 visible.
KV layouts follow that image's FlashMLA interface. Metadata, Q and indices are
fixed per shape; an optional cache flush is excluded from timed attention.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import time


def percentile(values, q):
    values = sorted(values)
    x = (len(values) - 1) * q
    low, high = math.floor(x), math.ceil(x)
    return values[low] + (values[high] - values[low]) * (x - low)


def cache(torch, batch, length, block, fmt):
    width = {'v41_fp8': 528, 'v41_nvfp4': 288}[fmt]
    pages = batch * math.ceil(length / block)
    raw = torch.empty((pages, block, 1, width), dtype=torch.uint8, device='cuda')
    flat = raw.view(pages, -1)
    if fmt == 'v41_fp8':
        data = flat[:, :block * 512].view(pages, block, 512)
        # Finite random values, not compressible all-zero KV.
        data.copy_(torch.randn(data.shape, device='cuda', dtype=torch.bfloat16).to(torch.float8_e4m3fn).view(torch.uint8))
        flat[:, block * 512:].fill_(127)  # E8M0 exponent for scale 1.
    else:
        flat[:, :block * 256].random_(0, 256)
        flat[:, block * 256:].fill_(56)  # E4M3 scale 1.
    return raw


def dequantize(torch, raw, indices, fmt):
    block = raw.shape[1]
    flat = raw.view(raw.shape[0], -1)
    pages = indices // block
    offsets = indices % block
    if fmt == 'v41_fp8':
        data = flat[:, :block * 512].view(-1, block, 512)
        return data[pages, offsets].contiguous().view(torch.float8_e4m3fn).float()
    packed = flat[:, :block * 256].view(-1, block, 256)[pages, offsets]
    nibbles = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
    table = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device='cuda')
    return table[nibbles]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lengths', type=int, nargs='+', default=[8192, 65536, 131072])
    p.add_argument('--batches', type=int, nargs='+', default=[1, 16, 48, 64])
    p.add_argument('--topk', type=int, default=512)
    p.add_argument('--heads', type=int, default=16, help='Per-rank query heads; 64/TP4 for the target model')
    p.add_argument('--kernel-heads', type=int, default=64, help='Audited FlashMLA FP8 decode supports 64/128 heads; the model pads smaller per-rank heads')
    p.add_argument('--extra-format', choices=['v41_fp8', 'v41_nvfp4'], default='v41_nvfp4')
    p.add_argument('--repeat', type=int, default=1)
    p.add_argument('--iterations', type=int, default=30)
    p.add_argument('--cache-mode', choices=['hot', 'cold'], default='cold')
    p.add_argument('--eager', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        raise RuntimeError('Refuse to overwrite an existing run')
    import torch
    import vllm
    from vllm.third_party.flashmla.flash_mla_interface import get_mla_metadata, flash_mla_with_kvcache
    if torch.cuda.device_count() != 1 or 'B200' not in torch.cuda.get_device_name():
        raise RuntimeError('Expose exactly one B200 for this standalone microbenchmark')
    torch.manual_seed(20261020)
    properties = torch.cuda.get_device_properties(0)
    result = {'status': 'running', 'platform': 'B200', 'scope': 'FlashMLA sparse attention kernel only; not Mega, full model, dataset replay or ASIC',
              'started_unix': time.time(), 'torch': torch.__version__, 'vllm': vllm.__version__,
              'device': str(properties), 'parameters': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, 'cases': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        tmp = args.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        tmp.replace(args.output)
    flush = torch.randint(0, 127, (512 * 1024 * 1024,), dtype=torch.int8, device='cuda') if args.cache_mode == 'cold' else None
    for length in args.lengths:
        if length < args.topk or length % 128:
            raise ValueError('Length must cover top-k and align to the 128-token block')
        for batch in args.batches:
            swa = cache(torch, batch, 128, 128, 'v41_fp8')
            extra = cache(torch, batch, length, 128, args.extra_format)
            if args.kernel_heads not in (64, 128) or not 0 < args.heads <= args.kernel_heads:
                raise ValueError('Invalid live/padded query heads')
            q = torch.zeros((batch, 1, args.kernel_heads, 512), device='cuda', dtype=torch.bfloat16)
            q[:, :, :args.heads].normal_()
            base = torch.arange(batch, device='cuda', dtype=torch.int32)[:, None] * length
            selected = torch.stack([torch.randperm(length, device='cuda', dtype=torch.int32)[:args.topk] for _ in range(batch)])
            extra_ids = (base + selected)[:, None, :].contiguous()
            swa_ids = (torch.arange(batch, device='cuda', dtype=torch.int32)[:, None] * 128 + torch.arange(128, device='cuda', dtype=torch.int32))[:, None, :].contiguous()
            metadata, _ = get_mla_metadata()
            def run():
                return flash_mla_with_kvcache(q, swa, None, None, 512, metadata,
                    is_fp8_kvcache=True, indices=swa_ids,
                    extra_k_cache=extra, extra_indices_in_kvcache=extra_ids)
            for _ in range(5):
                output, _ = run()
            torch.cuda.synchronize()
            if not torch.isfinite(output).all():
                raise AssertionError('Non-finite attention result')
            # Small shape validates packed layouts against the same q and K=V.
            reference_error = None
            if batch == 1:
                keys = torch.cat((dequantize(torch, swa, swa_ids[0, 0], 'v41_fp8'),
                                  dequantize(torch, extra, extra_ids[0, 0], args.extra_format)))
                scores = q[0, 0, :args.heads].float() @ keys.T / math.sqrt(512)
                reference = torch.softmax(scores, -1) @ keys
                reference_error = (output[0, 0, :args.heads].float() - reference).abs().max().item()
                if reference_error > 0.05:
                    raise AssertionError(f'KV-layout numeric validation failed: {reference_error}')
                del keys, scores, reference
            graph = None
            if not args.eager:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output, _ = run()
            timings = []
            for _ in range(args.iterations):
                if flush is not None:
                    flush.add_(1)
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                if graph is None:
                    output, _ = run()
                else:
                    graph.replay()
                end.record()
                end.synchronize()
                timings.append(start.elapsed_time(end))
            row = {'history_length': length, 'batch': batch, 'query_heads_per_rank': args.heads,
                   'kernel_query_heads': args.kernel_heads, 'head_padding_ratio': args.kernel_heads / args.heads,
                   'selected_compressed_tokens': args.topk, 'swa_tokens': 128,
                   'extra_format': args.extra_format, 'extra_record_bytes': extra.shape[-1],
                   'allocated_kv_bytes': swa.numel() + extra.numel(),
                   'logical_selected_kv_bytes_not_measured_dram': batch * (128 * 528 + args.topk * extra.shape[-1]),
                   'latency_ms_mean': statistics.fmean(timings), 'latency_ms_p50': percentile(timings, .5),
                   'latency_ms_p90': percentile(timings, .9), 'latency_ms_min': min(timings),
                   'latency_ms_max': max(timings), 'latency_samples_ms': timings,
                   'numeric_max_abs_error': reference_error, 'cache_mode': args.cache_mode,
                   'cuda_graph': not args.eager, 'finished_unix': time.time()}
            result['cases'].append(row)
            save()
            print(json.dumps({k: v for k, v in row.items() if k != 'latency_samples_ms'}), flush=True)
            del graph, run, output, q, swa, extra, swa_ids, extra_ids, selected, base, metadata
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    result.update(status='complete', finished_unix=time.time())
    save()


if __name__ == '__main__':
    main()
