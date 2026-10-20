#!/usr/bin/env python3
"""Generate publication figures from the released numeric evidence (CPU only)."""
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from analyze import ROOT, load, metrics

COLORS = {"v41": "#AF2E3D", "v4": "#E97F68", "qwen": "#CB5979"}
NAMES = {"v41": "DeepSeek-V4.1-Flash", "v4": "DeepSeek-V4-Flash-0731", "qwen": "Qwen3.8-27B"}
KEYS = list(NAMES)
FIG = ROOT / "figures"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none", "svg.hashsalt": "agent-workload-serving-study", "savefig.facecolor": "white"})


def finish(fig, name):
    for ax in fig.axes:
        ax.grid(alpha=.22, axis="y")
        ax.set_axisbelow(True)
    fig.savefig(FIG / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(FIG / (name + ".svg"), bbox_inches="tight", metadata={"Date": "2026-10-20"})
    svg = FIG / (name + ".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())+"\n", encoding="utf-8")
    plt.close(fig)


def pareto(replay):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.9))
    for key, name in NAMES.items():
        rows = sorted([r for r in replay["runs"] if r["model"] == name], key=lambda r: r["concurrency"])
        for ax, field in zip(axs, ("interactivity", "ttft")):
            xs = [metrics(r)["p90_interactivity"] if field == "interactivity" else r["ttft"]["p90"] / 1000 for r in rows]
            ys = [metrics(r)["total_tpgs"] for r in rows]
            ax.plot(xs, ys, color=COLORS[key], linewidth=1.8, label=name)
            for i, (x, y, r) in enumerate(zip(xs, ys, rows)):
                ax.plot(x, y, "o", ms=5, mfc=COLORS[key] if metrics(r)["sla_pass"] else "white", mec=COLORS[key], mew=1.6)
                if field == "interactivity":
                    offset = (0, 9) if key == "v41" else (0, -15 if key == "v4" else 10)
                    if key == "qwen" and r["concurrency"] in (32, 48): offset = (-10, -16)
                else:
                    offset = (-8, 9) if key == "v41" else (5, -14 if key == "v4" else 8)
                    if key == "v41" and r["concurrency"] == 16: offset = (10, 10)
                    if key == "qwen" and r["concurrency"] == 16: offset = (0, -15)
                ax.annotate(f'C{r["concurrency"]}', (x, y), xytext=offset, textcoords="offset points", fontsize=8, color=COLORS[key], ha="center")
    axs[0].axvline(20, color="#888888", ls="--", lw=1)
    axs[0].text(20.5, 27800, "TPOT = 50 ms", color="#777777", fontsize=9)
    axs[0].set_xlabel("P90 interactivity (decode_tokens/s/user)")
    axs[1].set_xlabel("P90 TTFT (s/request)")
    for ax in axs:
        ax.set_ylabel("Total TPGS (input + output tokens/s/GPU)")
        ax.set_ylim(16500, 29000)
    axs[0].set_title("Higher and farther right is better")
    axs[1].set_title("Higher and farther left is better")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .91), ncol=3, frameon=False)
    fig.suptitle("AgentX: measured throughput-latency trade-off", fontsize=16, y=1.01)
    fig.subplots_adjust(top=.75, bottom=.19, wspace=.3)
    fig.text(.06, .02, "B200 x4 / TP4 | max-num-seqs=256 | normal prefix cache | filled: both P90 SLAs pass; hollow: exceeded\nTotal TPGS includes cached input. Qwen skips 15/2048 candidates; cross-model effective workloads differ.", fontsize=9, color="#666666")
    finish(fig, "01_agentx_pareto")


def tasks(real):
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    status = {"passed": "PASS*", "self-reported": "UNVERIFIED", "stopped": "STOP"}
    for task, ax in zip(("task01", "task02", "task03", "task04"), axs.flat):
        for i, key in enumerate(KEYS):
            c = next(c for c in real["cases"] if c["modelKey"] == key and c["task"] == task)
            left = 0
            for field, hatch in [("apiMinutes", ""), ("toolMinutes", "///"), ("unclassifiedMinutes", "...")]:
                ax.barh(i, c[field], left=left, color=COLORS[key], edgecolor="white", linewidth=.6, hatch=hatch, height=.53, alpha=1 if field == "apiMinutes" else .65)
                left += c[field]
            label = f'{c["elapsedMinutes"]:.2f} min / {status[c["status"]]}'
            ax.text(c["elapsedMinutes"] + 3, i, label, va="center", fontsize=8)
        ax.set_title(task.replace("task", "Task "))
        ax.set_yticks(range(3), [NAMES[k] for k in KEYS], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 365)
        ax.set_xlabel("Elapsed minutes")
    fig.suptitle("Real tasks: disjoint observed time partition", fontsize=16)
    fig.legend(handles=[Patch(facecolor="#777777", label="API union"), Patch(facecolor="#aaaaaa", hatch="///", label="Tool outside API"), Patch(facecolor="#cccccc", hatch="...", label="Unclassified")], loc="upper center", bbox_to_anchor=(.5, .94), ncol=3, frameon=False)
    fig.subplots_adjust(top=.82, bottom=.15, left=.16, hspace=.6, wspace=.4)
    fig.text(.07, .02, "API is not pure GPU time. Tool-total union may overlap API and is listed separately in the tables.\n*Task 3: full C1-C7 acceptance; Task 4: saved-artifact C1-C6 only. V4 Task 3/4 are legacy stops beyond 200 min.", fontsize=9, color="#666666")
    finish(fig, "02_real_task_time")


def calls_tokens(real):
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    for i, key in enumerate(KEYS):
        cs = [c for c in real["cases"] if c["modelKey"] == key]
        for j, (field, hatch) in enumerate([("modelCalls", ""), ("toolCalls", "///")]):
            val = sum(c[field] for c in cs)
            axs[0].bar(i + (j-.5)*.32, val, .3, color=COLORS[key], hatch=hatch, edgecolor="white")
            axs[0].text(i + (j-.5)*.32, val+16, str(val), ha="center", fontsize=9)
        values = [sum(c["cachedTokens"] for c in cs), sum(c["inputTokens"]-c["cachedTokens"] for c in cs), sum(c["outputTokens"] for c in cs)]
        left = 0
        for val, hatch in zip(values, ("///", "", "...")):
            axs[1].barh(i, val/1e6, left=left, color=COLORS[key], edgecolor="white", hatch=hatch, height=.55)
            left += val/1e6
        axs[1].text(left+.6, i, f'{left:.2f} M', va="center", fontsize=9)
    labels = ["V4.1", "V4-0731", "Qwen3.8-27B"]
    axs[0].set_xticks(range(3), labels)
    axs[0].set_ylabel("Calls across four tasks")
    axs[0].legend(handles=[Patch(color="#777777", label="Model calls"), Patch(facecolor="#777777", hatch="///", label="Tool calls")], frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=2, fontsize=9)
    axs[0].set_ylim(0, 1080)
    axs[1].set_yticks(range(3), labels)
    axs[1].invert_yaxis()
    axs[1].set_xlim(0, 90)
    axs[1].set_xlabel("Cumulative input + output (million tokens)")
    axs[1].legend(handles=[Patch(facecolor="#777777", hatch="///", label="Cached input"), Patch(color="#777777", label="Uncached input"), Patch(facecolor="#777777", hatch="...", label="Output")], frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=3, fontsize=8)
    fig.suptitle("Real task workload volume: paths and completion states differ", fontsize=15)
    fig.subplots_adjust(top=.78, bottom=.18, wspace=.35)
    fig.text(.08, .02, "Cached tokens are a subset of input. Auxiliary calls are already included. Stopped tasks are retained; this is not a cost or quality ranking.", fontsize=9, color="#666666")
    finish(fig, "03_calls_tokens")


def cache_events(real):
    selected = [("v41", "task03", 41), ("v4", "task03", 159), ("qwen", "task01", 143)]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, (key, task, idx) in zip(axs, selected):
        c = next(c for c in real["cases"] if c["modelKey"] == key and c["task"] == task)
        rows = [r for r in c["requests"] if idx-3 <= r["index"] <= idx+3]
        ax.plot([r["index"] for r in rows], [100*r["cachedTokens"]/r["inputTokens"] for r in rows], "o-", ms=4, lw=1.6, color=COLORS[key])
        row = next(r for r in rows if r["index"] == idx)
        pct = 100*row["cachedTokens"]/row["inputTokens"]
        ax.axvline(idx, color="#9ca3af", ls="--", lw=1)
        ax.text(idx+1.4, 35, f'Notification\n{pct:.2f}%\n{row["cachedTokens"]:,} cached', fontsize=8.5, color=COLORS[key])
        ax.set_title(NAMES[key] + " / " + task, fontsize=10)
        ax.set_xlabel("Request index (1-based)")
        ax.set_ylim(-3, 110)
        ax.set_xticks([r["index"] for r in rows][::2])
    axs[0].set_ylabel("Request cache-hit (%)")
    fig.suptitle("New inline system notifications coincide with prefix-cache valleys", fontsize=15)
    fig.subplots_adjust(bottom=.24, top=.78, wspace=.23)
    fig.text(.06, .03, "Old history, top-level system and tools stay unchanged. Installed adapter source plus offline counterfactual confirms front-merging.\nThese examples are not compaction boundaries; not every cache valley is a notification.", fontsize=9, color="#666666")
    finish(fig, "04_notification_valleys")


def graph_ablation(graph):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, title in zip(axs, ("client_ttft_ms", "server_scheduled_to_first_token_ms"), ("Client TTFT (includes transport)", "Scheduled-to-first-token (excludes queue)")):
        for i, key in enumerate(("agentx", "mongo")):
            g = graph[key]
            for j, config in enumerate(("before_paired", "after_paired")):
                v = g[config][metric]["mean"]
                ax.bar(i+(j-.5)*.32, v, .3, color=("#AF2E3D", "#E97F68")[j])
                ax.text(i+(j-.5)*.32, v+20, f"{v:.2f}", ha="center", fontsize=9)
        ax.set_xticks([0, 1], ["Public / 112 paired", "Internal / 100 paired"])
        ax.set_ylabel("Mean ms")
        ax.set_ylim(0, max(graph[k]["after_paired"][metric]["mean"] for k in ("agentx", "mongo"))*1.23)
        ax.set_title(title, fontsize=11)
    fig.suptitle("V4.1 CUDA Graph ablation / C4 / normal prefix cache", fontsize=15)
    fig.legend(handles=[Patch(color="#AF2E3D", label="FULL_AND_PIECEWISE"), Patch(color="#E97F68", label="FULL_DECODE_ONLY")], loc="upper center", bbox_to_anchor=(.5, .93), ncol=2, frameon=False)
    fig.subplots_adjust(top=.75, bottom=.2, wspace=.25)
    fig.text(.08, .02, "Means over matched requests; not repeated-run confidence intervals. Server interval excludes initial queue, not inter-iteration interference.", fontsize=9, color="#666666")
    finish(fig, "05_graph_ablation")


def stability():
    base, zero = load("cache_baseline.json"), load("cache_stability.json")
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.8))
    for ax, key in zip(axs, ("public", "internal")):
        normal = base["datasets"][key]
        ys = [normal["v4_ttft_ms"], normal["v41_ttft_ms"]]
        ax.plot([0, 1], ys, "o-", ms=3, color="#CB5979", label="Normal cache / historical single run")
        rows = [zero["datasets"][key][n] for n in ("V4 Decode-only", "V4.1 Piecewise")]
        zm = [r["ttft_mean_ms"] for r in rows]
        ax.errorbar([0, 1], zm, yerr=[[r["ttft_mean_ms"]-r["ttft_min_ms"] for r in rows], [r["ttft_max_ms"]-r["ttft_mean_ms"] for r in rows]], fmt="o-", ms=3, color="#AF2E3D", ecolor="#DEA1BE", capsize=5, label="Zero cache / 3-run mean and min-max")
        for i, y in enumerate(ys): ax.annotate(f"{y:.2f}", (i, y), xytext=(0, -20), textcoords="offset points", ha="center", color="#CB5979")
        for i, y in enumerate(zm): ax.text(i, rows[i]["ttft_max_ms"]+280, f"{y:.2f}", ha="center", color="#AF2E3D")
        delta = zm[0]-zm[1]
        ds = [a["ttft_mean_ms"]-b["ttft_mean_ms"] for a,b in zip(rows[0]["runs"], rows[1]["runs"])]
        ax.text(.5, 3200, f'Zero-cache reduction: {delta:.2f} ms\nround-label deltas: [{min(ds):.2f}, {max(ds):.2f}]', ha="center", fontsize=9, color="#AF2E3D")
        ax.set_title("Public / small subset" if key == "public" else "Internal / 100 requests")
        ax.set_xticks([0, 1], ["V4-0731", "V4.1"])
        ax.set_xlim(-.22, 1.22)
        ax.set_ylim(0, 7700)
        ax.set_ylabel("Mean client TTFT (ms)")
    fig.suptitle("Cache-condition comparison: absolute deltas and observed variation", fontsize=15)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .94), ncol=1, frameon=False)
    fig.subplots_adjust(top=.71, bottom=.18, wspace=.24)
    fig.text(.07, .02, "Historical C4 study, not the 2048-candidate capacity experiment. Min-max is not a confidence interval.\nInternal zero-cache output lengths differ across runs; the normal-cache condition has no repeat distribution.", fontsize=9, color="#666666")
    finish(fig, "06_cache_condition")


def resources(real):
    fig, axs = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    for r in real["resources"]:
        for ax, field in zip(axs, ("gpu", "kv")):
            bins = {}
            for t,v in r[field]: bins.setdefault(math.floor(t), []).append(v)
            xs = list(range(min(bins), max(bins)+1))
            ys = [(sum(bins[t])/len(bins[t]) if field == "gpu" else max(bins[t])) if t in bins else math.nan for t in xs]
            ax.plot([t+.5 for t in xs], ys, color=COLORS[r["modelKey"]], label=NAMES[r["modelKey"]], lw=1.2, alpha=.85)
    axs[0].set_ylabel("GPU utilization (%)\n1-min sample mean", labelpad=8)
    axs[1].set_ylabel("Allocated KV blocks (%)\n1-min peak", labelpad=8)
    axs[1].set_xlabel("Minutes since each model's four-client batch began")
    axs[0].set_ylim(0, 105)
    axs[1].set_ylim(0, max(v for r in real["resources"] for _,v in r["kv"])*1.2)
    fig.suptitle("Shared model service resource timelines (descriptive only)", fontsize=14)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .93), ncol=3, frameon=False, fontsize=9)
    fig.subplots_adjust(top=.82, bottom=.19, hspace=.2)
    fig.text(.08, .02, "Different experiments aligned by elapsed time. Four tasks share a model service. Raw samples remain in the released data.\nDisplay uses 1-min sample means (GPU) / peaks (KV); missing bins are not interpolated. GPU utilization is not HBM bandwidth.\nAllocated KV blocks do not measure all resident reusable cache. Sampling/clock limits apply.", fontsize=9, color="#666666")
    finish(fig, "07_resource_timelines")


def notification_control():
    rows = load("notification_control.json")["results"][1:]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    labels = ["Before notification\nrepeat / warm", "New history\nnotification removed", "Same new history\nnotification added", "Identical notified\nrequest repeated", "Notification\nremoved again"]
    bars = ax.bar(range(5), [r["cache_hit_percent"] for r in rows], color=["#BDA9C4", "#CB5979", "#AF2E3D", "#E97F68", "#CB5979"], width=.6)
    for b,r in zip(bars, rows): ax.text(b.get_x()+b.get_width()/2, b.get_height()+3, f'{r["cache_hit_percent"]:.2f}%\n{r["cached_tokens"]:,} cached', ha="center", fontsize=9)
    ax.set_ylim(0, 119)
    ax.set_xticks(range(5), labels, fontsize=9)
    ax.set_ylabel("Request cache-hit (%)")
    ax.set_title("Latest-client actual-request control / Claude Code 2.1.280", fontsize=14)
    fig.subplots_adjust(bottom=.26)
    fig.text(.09, .02, "Separate gateway deployment; output capped at 16 only for this behavioral check. This is not a throughput benchmark.\nFunctional cache loss is reproduced; actual A800-route converter source and physical hardware were not independently audited.", fontsize=9, color="#666666")
    finish(fig, "08_latest_client_control")


def main():
    FIG.mkdir(exist_ok=True)
    replay, real, graph = load("agentx.json"), load("real_tasks.json"), load("cuda_graph_ablation.json")
    pareto(replay)
    tasks(real)
    calls_tokens(real)
    cache_events(real)
    graph_ablation(graph)
    stability()
    resources(real)
    notification_control()
    print(json.dumps({"figures": 8, "formats": ["png", "svg"], "directory": str(FIG)}))


if __name__ == "__main__":
    main()
