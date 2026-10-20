#!/usr/bin/env python3
"""Rebuild report tables and machine-readable findings from released numbers."""
from collections import Counter
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("v41", "v4", "qwen")


def load(name):
    return json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))


def metrics(r):
    return {
        "request_throughput": r["requestCount"] / r["elapsedSeconds"],
        "output_token_throughput": r["outputTokens"] / r["elapsedSeconds"],
        "total_token_throughput": (r["promptTokens"] + r["outputTokens"]) / r["elapsedSeconds"],
        "total_tpgs": (r["promptTokens"] + r["outputTokens"]) / r["elapsedSeconds"] / r["gpuCount"],
        "output_tpgs": r["outputTokens"] / r["elapsedSeconds"] / r["gpuCount"],
        "p90_interactivity": 1000 / r["tpot"]["p90"],
        "sla_pass": r["ttft"]["p90"] <= 10000 and r["tpot"]["p90"] <= 50,
    }


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
                     + ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]) + "\n"


def distribution(d, decimals, scale=1):
    return "/".join(f"{d[k] / scale:.{decimals}f}" for k in ("p50", "p90", "p99", "max"))


def main():
    replay, real, graph = load("agentx.json"), load("real_tasks.json"), load("cuda_graph_ablation.json")
    out = ROOT / "tables"
    out.mkdir(exist_ok=True)
    runs = [{**r, **metrics(r)} for r in replay["runs"]]
    headers = ["Model", "C", "Requests", "TTFT P90(s)", "TPOT P90(ms)", "P90 interactivity", "Total TPGS", "Output tok/s", "Cache-hit(%)", "SLA"]
    rows = [[r["model"], r["concurrency"], r["requestCount"], f'{r["ttft"]["p90"]/1000:.3f}', f'{r["tpot"]["p90"]:.2f}', f'{r["p90_interactivity"]:.2f}', f'{r["total_tpgs"]:.2f}', f'{r["output_token_throughput"]:.2f}', f'{r["cacheHitPercent"]:.2f}', "通过" if r["sla_pass"] else "超限"] for r in runs]
    (out / "agentx_summary.md").write_text("# AgentX 并发结果\n\n" + table(headers, rows) + "\nTTFT 和 TPOT 为请求级 P90；Cache-hit 为 token 加权全局值。SLA 同时要求 P90 TTFT≤10 s、P90 TPOT≤50 ms。吞吐为整体运行值。\n", encoding="utf-8")
    fields = ["model", "concurrency", "candidateRequests", "requestCount", "skippedOverlength", "elapsedSeconds", "promptTokens", "outputTokens", "cachedPromptTokens", "cacheHitPercent", "request_throughput", "output_token_throughput", "total_token_throughput", "total_tpgs", "output_tpgs", "p90_interactivity", "sla_pass"]
    with (out / "agentx_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in runs)
    rows = [[r["model"], r["concurrency"], distribution(r["isl"], 0), distribution(r["osl"], 0), distribution(r["ttft"], 3, 1000), distribution(r["tpot"], 2), distribution(r["cacheHit"], 2)] for r in runs]
    (out / "agentx_percentiles.md").write_text("# AgentX 分位数\n\n" + table(["Model", "C", "ISL", "OSL", "TTFT(s)", "TPOT(ms)", "请求 Cache-hit(%)"], rows) + "\n每格依次为 P50/P90/P99/Max；ISL/OSL 展示为整数，未改写原始插值结果。请求缓存百分比分布不等于全局 token 加权命中率。\n", encoding="utf-8")
    rows = [[name, r["contextLimit"], r["candidateRequests"], r["requestCount"], r["skippedOverlength"], r["sessionCount"], distribution(r["isl"], 0), distribution(r["osl"], 0)] for name in replay["models"].values() for r in runs if r["model"] == name and r["concurrency"] == 16]
    (out / "dataset.md").write_text("# 有效数据集\n\n" + table(["Model", "上下文限制", "候选请求", "有效请求", "跳过", "有效会话", "ISL", "OSL"], rows) + "\nISL/OSL 每格依次为 P50/P90/P99/Max。超上下文样本未计入分布。\n", encoding="utf-8")
    status = {"passed": "独立验收通过", "stopped": "中止/未完成", "self-reported": "自报完成/隐藏未验"}
    rows = [[real["models"][c["modelKey"]], c["task"], status[c["status"]], f'{c["elapsedMinutes"]:.2f}', f'{c["apiMinutes"]:.2f} ({100*c["apiMinutes"]/c["elapsedMinutes"]:.2f}%)', f'{c["toolTotalMinutes"]:.2f} ({100*c["toolTotalMinutes"]/c["elapsedMinutes"]:.2f}%)', f'{c["toolMinutes"]:.2f}', f'{c["cacheHitPercent"]:.2f}'] for c in real["cases"]]
    (out / "real_tasks.md").write_text("# 真实任务耗时\n\n" + table(["Model", "Task", "状态", "总耗时(min)", "API并集 min(占比)", "工具总并集 min(占比)", "工具不与API重叠(min)", "全局 Cache-hit(%)"], rows) + "\n占比以总耗时为分母。API 与工具总并集可以重叠，不能直接相加。V4 任务3/4是新增限时前的历史中止，实际约264.35分钟；其余上限200分钟。独立验收耗时另计。\n", encoding="utf-8")
    rows = [[real["models"][c["modelKey"]], c["task"], c["modelCalls"], c["toolCalls"], c["auxiliaryCalls"], c["toolErrorFlags"], c["inputTokens"], c["cachedTokens"], c["outputTokens"]] for c in real["cases"]]
    (out / "calls_tokens.md").write_text("# 调用与 Token 计数\n\n" + table(["Model", "Task", "模型调用", "工具调用", "辅助调用", "工具错误标记", "输入", "缓存输入", "输出"], rows) + "\n缓存输入是输入的子集；辅助调用已包含于模型调用中，不重复相加。工具错误标记不等于测试失败总数。\n", encoding="utf-8")
    events = {}
    for key in KEYS:
        cs = [c for c in real["cases"] if c["modelKey"] == key]
        cnt = Counter(e["kind"] for c in cs for e in c["cacheEvents"])
        events[key] = {"compaction": sum(len(c["compactions"]) for c in cs), **dict(cnt)}
    facts = {
        "replay_runs": len(runs), "real_cases": len(real["cases"]), "verified_requests": real["verifiedRequests"],
        "highest_tested_passing": {name: max(r["concurrency"] for r in runs if r["model"] == name and r["sla_pass"]) for name in replay["models"].values()},
        "status_counts": dict(Counter(c["status"] for c in real["cases"])), "cache_events": events,
        "api_share_range_percent": [min(100*c["apiMinutes"]/c["elapsedMinutes"] for c in real["cases"]), max(100*c["apiMinutes"]/c["elapsedMinutes"] for c in real["cases"])],
        "tool_outside_api_share_range_percent": [min(100*c["toolMinutes"]/c["elapsedMinutes"] for c in real["cases"]), max(100*c["toolMinutes"]/c["elapsedMinutes"] for c in real["cases"])],
        "graph_paired_server_delta_ms": {k: graph[k]["paired_after_minus_before"]["scheduled_to_first_token_ms"]["mean"] for k in ("agentx", "mongo")},
    }
    (out / "findings.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(facts, ensure_ascii=False))


if __name__ == "__main__":
    main()
