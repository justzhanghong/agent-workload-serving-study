#!/usr/bin/env python3
"""Export allowlisted measurements, never original task text or deployment paths.

This is an offline export. It neither runs models nor modifies its inputs.
Inputs are existing audited summaries and numeric pairing files, not API payloads.
"""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path


RUN_FIELDS = "model dataset cacheMode concurrency device gpuCount candidateRequests requestCount skippedOverlength sessionCount contextLimit elapsedSeconds promptTokens outputTokens cachedPromptTokens ttft tpot isl osl cacheHit cacheHitPercent tpotExcluded".split()
CASE_FIELDS = "id modelKey task status elapsedMinutes apiMinutes toolMinutes unclassifiedMinutes toolTotalMinutes modelCalls toolCalls auxiliaryCalls toolErrorFlags inputTokens cachedTokens outputTokens cacheHitPercent limitMinutes legacyStop acceptanceMinutes acceptanceScope exitCodeCaptured compactions requests cacheEvents".split()
MODELS = {"v41": "DeepSeek-V4.1-Flash", "v4": "DeepSeek-V4-Flash-0731", "qwen": "Qwen3.8-27B"}


def select(row, fields):
    return {k: row[k] for k in fields}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("replay", "real", "graph", "stability", "notification_control"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache-baseline-plot", type=Path, required=True)
    p.add_argument("--micro-grid", type=Path, required=True)
    p.add_argument("--graph-paired-directory", type=Path, required=True)
    args = p.parse_args()
    originals, source_hashes = {}, {}
    for name in ("replay", "real", "graph", "stability", "notification_control"):
        body = getattr(args, name).read_bytes()
        originals[name] = json.loads(body)
        source_hashes[name] = hashlib.sha256(body).hexdigest()
    micro_body = args.micro_grid.read_bytes()
    micro = json.loads(micro_body)
    source_hashes["micro_grid"] = hashlib.sha256(micro_body).hexdigest()
    dump(args.output / "micro_grid.json", select(micro, ["method", "server_validation", "direct_v4_vs_v41", "v41_ablations"]))
    paired = {}
    for key in ("agentx", "mongo"):
        pair_body = (args.graph_paired_directory / (key + "_paired_per_request.csv")).read_bytes()
        source_hashes["graph_" + key + "_pairs"] = hashlib.sha256(pair_body).hexdigest()
        rows = list(csv.DictReader(pair_body.decode().splitlines()))
        names = [prefix + field for prefix in ("before_", "after_", "delta_after_minus_before_")
                 for field in ("input_tokens", "cached_tokens", "uncached_tokens", "cache_hit_rate_pct", "ttft_ms", "queue_ms", "client_ttft_minus_queue_ms", "scheduled_to_first_token_ms", "client_minus_server_prefill_ms")]
        paired[key] = [{"pair": i+1, **{name: float(row[name]) for name in names}} for i,row in enumerate(rows)]
    dump(args.output / "graph_paired.json", paired)
    plot_body = args.cache_baseline_plot.read_bytes()
    tree = ast.parse(plot_body)
    baseline = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "DATA" for t in n.targets))
    source_hashes["historical_cache_plot"] = hashlib.sha256(plot_body).hexdigest()
    dump(args.output / "cache_baseline.json", {
        "concurrency": 4, "condition": "normal_prefix_cache", "repeats": 1,
        "datasets": {key: {"v4_ttft_ms": row["normal"]["v4"], "v41_ttft_ms": row["normal"]["v41"]}
                     for key, row in baseline.items()},
        "scope": "Historical small-subset chart values, not the later 2048-candidate capacity study. No normal-cache repeat distribution was collected.",
    })
    replay = originals["replay"]
    runs = []
    for r in replay["runs"]:
        row = select(r, RUN_FIELDS)
        if row["model"] == "DeepSeek-V4-Flash":
            row["model"] = MODELS["v4"]
        row["evidence"] = {k: r["source"][k] for k in ("requestsSha256", "manifestSha256")}
        runs.append(row)
    runs.sort(key=lambda r: (list(MODELS.values()).index(r["model"]), r["concurrency"]))
    dump(args.output / "agentx.json", {"schemaVersion": 1, "models": MODELS, "runs": runs})
    real = originals["real"]
    cases = [select(c, CASE_FIELDS) for c in real["cases"]]
    # Request/event/resource objects in this export have strictly numeric or enum
    # fields; metadata (checkpoints, source paths, request IDs) is not copied.
    for c in cases:
        c["requests"] = [select(r, "index minutes inputTokens cachedTokens outputTokens ttftSeconds tpotMs auxiliary".split()) for r in c["requests"]]
        c["cacheEvents"] = [select(e, ["index", "kind"]) for e in c["cacheEvents"]]
    resources = [select(r, "modelKey durationMinutes intervals gpu kv cpuIo".split()) for r in real["resources"]]
    dump(args.output / "real_tasks.json", {"schemaVersion": 1, "models": MODELS, "verifiedRequests": real["source"]["verifiedRequests"], "cases": cases, "resources": resources, "reuse": real["reuse"]})
    graph = originals["graph"]
    dump(args.output / "cuda_graph_ablation.json", select(graph, ["semantics", "agentx", "mongo"]))
    stability = originals["stability"]
    dump(args.output / "cache_stability.json", select(stability, ["condition", "concurrency", "aggregation", "datasets", "normal_cache_queue_reference_ms"]))
    control = originals["notification_control"]
    dump(args.output / "notification_control.json", {
        "clientVersion": control["source_client_version"],
        "maxOutputTokens": control["max_output_tokens_per_request"],
        "generatedToolsNotExecuted": control["generated_tools_not_executed"],
        "results": [select(r, ["label", "prompt_tokens_api_sum", "cached_tokens", "cache_hit_percent"]) for r in control["results"]],
    })
    # File content hashes support owner-side audit without leaking source paths.
    dump(args.output / "provenance.json", {
        "schemaVersion": 1, "assembledDate": "2026-10-20", "sourceFileSha256": source_hashes,
        "originalRealEvidenceSha256": sorted(set(real["source"]["sha256"].values())),
        "publicationPolicy": "Numeric measurements and content fingerprints only; raw prompts, code submissions, tool output, account data and deployment paths omitted.",
        "scope": "Offline reanalysis of existing experiments; no new inference runs.",
    })
    print(json.dumps({"runs": len(runs), "cases": len(cases), "requests": sum(len(c["requests"]) for c in cases), "output": str(args.output)}))


if __name__ == "__main__":
    main()
