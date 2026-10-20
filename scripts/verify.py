#!/usr/bin/env python3
"""Verify released-data arithmetic, report citations, links and publication scope.

This checks internal consistency, not independent access to withheld evidence.
"""
from collections import Counter
from datetime import date
import hashlib
import ipaddress
import json
import math
from pathlib import Path
import re
import statistics
from analyze import ROOT, load, metrics

CHECKS = []
EXCLUDED = {".git", "__pycache__", ".venv", ".delivery"}


def require(condition, name):
    if not condition:
        raise AssertionError(name)
    CHECKS.append(name)


def close(a, b, name):
    require(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-6), name)


def percentile(vals, q):
    vals = sorted(vals)
    x = (len(vals)-1)*q
    a,b = math.floor(x), math.ceil(x)
    return vals[a] + (vals[b]-vals[a])*(x-a)


def main():
    replay, real = load("agentx.json"), load("real_tasks.json")
    report = (ROOT / "paper/report.zh-CN.md").read_text(encoding="utf-8")
    require(len(replay["runs"]) == 15, "15 replay points")
    for model in replay["models"].values():
        rs = [r for r in replay["runs"] if r["model"] == model]
        require(sorted(r["concurrency"] for r in rs) == [16,32,48,56,64], model+" grid")
        require(len({tuple(r[k] for k in ("requestCount", "skippedOverlength", "sessionCount", "promptTokens", "outputTokens", "contextLimit")) for r in rs}) == 1, model+" fixed workload")
        require(len({r["evidence"]["manifestSha256"] for r in rs}) == 1, model+" fixed manifest")
        best = max(r["concurrency"] for r in rs if metrics(r)["sla_pass"])
        require(best == (56 if model == replay["models"]["v41"] else 48), model+" measured SLA boundary")
    aliases = dict(zip(replay["models"].values(), ("V4.1", "V4", "Qwen")))
    for r in replay["runs"]:
        ident = r["model"]+" C"+str(r["concurrency"])
        require(r["requestCount"]+r["skippedOverlength"] == 2048, ident+" candidates")
        require(r["gpuCount"] == 4 and r["cacheMode"] == "normal", ident+" hardware/cache")
        require(r["tpotExcluded"] == 3, ident+" TPOT exclusions")
        close(r["cacheHitPercent"], 100*r["cachedPromptTokens"]/r["promptTokens"], ident+" global cache accounting")
        for field in ("ttft", "tpot", "isl", "osl", "cacheHit"):
            v = r[field]
            require(v["min"] <= v["p50"] <= v["p90"] <= v["p99"] <= v["max"], ident+" "+field+" order")
        require(r["isl"]["max"] <= r["contextLimit"], ident+" context")
        expected = f'| {aliases[r["model"]]} | {r["concurrency"]} | {r["ttft"]["p90"]/1000:.3f} | {r["tpot"]["p90"]:.2f} | {metrics(r)["total_tpgs"]:.2f} | {r["cacheHitPercent"]:.2f} | {"通过" if metrics(r)["sla_pass"] else "超限"} |'
        require(expected in report, ident+" report row")
    require(len(real["cases"]) == 12, "12 task cases")
    require(Counter(c["status"] for c in real["cases"]) == {"passed":3,"self-reported":6,"stopped":3}, "acceptance states")
    require(sum(c["modelCalls"] for c in real["cases"]) == real["verifiedRequests"] == 2349, "2349 formal requests")
    require(sum(c["auxiliaryCalls"] for c in real["cases"]) == 21, "21 auxiliary calls")
    require(sum(c["toolErrorFlags"] for c in real["cases"]) == 129, "129 tool error flags")
    events = Counter()
    for c in real["cases"]:
        ident = c["id"]
        close(c["elapsedMinutes"], c["apiMinutes"]+c["toolMinutes"]+c["unclassifiedMinutes"], ident+" disjoint elapsed partition")
        require(c["toolTotalMinutes"] >= c["toolMinutes"], ident+" overlapping tool total")
        require(len(c["requests"]) == c["modelCalls"], ident+" request count")
        require([r["index"] for r in c["requests"]] == list(range(1,c["modelCalls"]+1)), ident+" request indices")
        for field, total in (("inputTokens","inputTokens"),("cachedTokens","cachedTokens"),("outputTokens","outputTokens")):
            require(sum(r[field] for r in c["requests"]) == c[total], ident+" "+field+" reconciliation")
        close(c["cacheHitPercent"], 100*c["cachedTokens"]/c["inputTokens"], ident+" weighted cache")
        require(sum(r["auxiliary"] for r in c["requests"]) == c["auxiliaryCalls"], ident+" auxiliary subset")
        if c["legacyStop"]:
            require(c["modelKey"] == "v4" and c["task"] in ("task03","task04") and c["elapsedMinutes"] > 264, ident+" legacy stop not clamped")
        for event in c["cacheEvents"]:
            require(event["index"] not in c["compactions"], ident+" event disjoint from compaction")
            prev, cur = c["requests"][event["index"]-2:event["index"]]
            drop = 100*prev["cachedTokens"]/prev["inputTokens"] - 100*cur["cachedTokens"]/cur["inputTokens"]
            if event["kind"] == "system-prefix-change":
                require(drop >= 30 and 100*cur["cachedTokens"]/cur["inputTokens"] < 50, ident+" deep valley rule")
            else:
                require(event["kind"] == "input-growth" and drop >= 15 and cur["cachedTokens"] >= prev["cachedTokens"] and cur["inputTokens"] > prev["inputTokens"], ident+" denominator rule")
            events[event["kind"]] += 1
        events["compaction"] += len(c["compactions"])
    require(events == {"compaction":20,"system-prefix-change":46,"input-growth":28}, "94 annotated events")
    require(sum(r["withinCount"] for r in real["reuse"]) == 18 and sum(r["crossCount"] for r in real["reuse"]) == 0, "certified read-only subset")
    close(sum(r["withinSeconds"] for r in real["reuse"]), 1.1252665519714355, "certified read seconds")
    graph, paired = load("cuda_graph_ablation.json"), load("graph_paired.json")
    for key, count, outliers in (("agentx",112,11),("mongo",100,14)):
        rows, summary = paired[key], graph[key]
        require(len(rows) == count, key+" graph paired count")
        for field in ("input_tokens","cached_tokens","uncached_tokens","ttft_ms","queue_ms","client_ttft_minus_queue_ms","scheduled_to_first_token_ms","client_minus_server_prefill_ms"):
            for r in rows:
                close(r["after_"+field]-r["before_"+field], r["delta_after_minus_before_"+field], key+" graph difference "+field)
        vals = [r["delta_after_minus_before_scheduled_to_first_token_ms"] for r in rows]
        dst = summary["paired_after_minus_before"]["scheduled_to_first_token_ms"]
        close(statistics.fmean(vals), dst["mean"], key+" graph mean")
        close(percentile(vals,.5), dst["p50"], key+" graph paired P50")
        close(percentile(vals,.9), dst["p90"], key+" graph paired P90")
        require(sum(v > 1000 for v in vals) == outliers, key+" graph >1s subset")
        for period in ("before", "after"):
            close(statistics.fmean(r[period+"_scheduled_to_first_token_ms"] for r in rows), summary[period+"_paired"]["server_scheduled_to_first_token_ms"]["mean"], key+" graph "+period+" execution mean")
    zero = load("cache_stability.json")
    for key, mean, limits in (("public",920.28,(667.61,1197.08)),("internal",800.09,(159.00,1458.27))):
        a,b = [zero["datasets"][key][name] for name in ("V4 Decode-only","V4.1 Piecewise")]
        require([r["run"] for r in a["runs"]] == [r["run"] for r in b["runs"]], key+" repeat labels")
        ds = [x["ttft_mean_ms"]-y["ttft_mean_ms"] for x,y in zip(a["runs"],b["runs"])]
        require(round(statistics.fmean(ds),2) == mean and (round(min(ds),2),round(max(ds),2)) == limits, key+" observed delta range")
        require(all(r["cached_tokens"] == 0 for r in a["runs"]+b["runs"]), key+" actual zero hits")
    micro = load("micro_grid.json")
    require(len(micro["v41_ablations"]) == 9, "9 synthetic shapes retained")
    for row in micro["v41_ablations"]:
        close(row["piecewise_graph_benefit_cached_ms"]-row["piecewise_graph_benefit_zero_ms"], row["piecewise_graph_cache_specific_ms"], "micro-grid difference of differences")
    controls = load("notification_control.json")["results"]
    require(len(controls) == 6, "6 notification control calls")
    require([r["cached_tokens"] for r in controls[2:]] == [2560,0,2560,2560], "remove/add/repeat/remove cache control")
    for r in controls:
        close(r["cache_hit_percent"], 100*r["cached_tokens"]/r["prompt_tokens_api_sum"], "notification control cache arithmetic")
    provenance = load("provenance.json")
    require(len(provenance["sourceFileSha256"]) == 9 and all(re.fullmatch("[a-f0-9]{64}", x) for x in provenance["sourceFileSha256"].values()), "9 input file fingerprints")
    literature = load("literature.json")
    works = literature["works"]
    refs = [work["ref"] for work in works]
    grouped = [ref for group in literature["focusGroups"] for ref in group["refs"]]
    require(len(works) == len(set(refs)) == 22, "22 unique workload/core and background entries")
    require(sum(w["scope"] == "core" for w in works) == literature["coreWorks"] == 18, "18 workload/system core studies and tools")
    require(sum(w["scope"] == "scenario-background" for w in works) == literature["backgroundWorks"] == 4, "4 scenario-only background works")
    require(sorted(grouped) == sorted(refs), "literature focus groups cover each work once")
    require(refs.count(1) == 1 and 2 not in refs, "papers.cool discovery not double counted")
    require([group["name"] for group in literature["focusGroups"]][:2] == ["多轮负载特征与生产轨迹","轮间KV状态与缓存管理"], "workload characterization and state lead review")
    require(set(map(int,re.findall(r"^\[(\d+)\] ",report,re.M))) == set(range(1,27)), "26 report references including documentation")
    index = (ROOT/"docs/literature-index.md").read_text(encoding="utf-8")
    bib = (ROOT/"references.survey.bib").read_text(encoding="utf-8")
    for work in works:
        require(work["source"].startswith("https://") and work["source"] in index, work["name"]+" linked primary source")
        require(all(work.get(k) for k in ("title","authors","method","measure","boundary")), work["name"]+" review fields")
        if work["ref"] >= 8:
            require(f'@misc{{surveyRef{work["ref"]},' in bib and work["title"] in report, work["name"]+" report and BibTeX")
    for name in ("TraceLab","Copilot","CacheWise","Continuum","Autellix","HEAR","AgentSysBench","InferCept","CPU","KV"):
        require(name in report.split("## 3. 方法")[0], name+" in focused related work")
    readme = (ROOT/"README.md").read_text(encoding="utf-8")
    featured = [w for w in works if w.get("readmeFeatured")]
    featured_text = readme.split("### 直接相关的Workload研究",1)[1].split("![等待期间",1)[0]
    featured_rows = [line for line in featured_text.splitlines() if line.startswith("| [")]
    require(len(featured) == len(featured_rows) == 12, "12 separately dated featured workload works")
    require("| 代表工作 | 日期 |" in featured_text, "featured workload table date column")
    for work in featured:
        require(date.fromisoformat(work["publicationDate"]).isoformat() == work["publicationDate"], work["name"]+" ISO publication date")
        require(work["dateBasis"] in ("arxiv-v1-submission", "official-announcement") and work["dateSource"].startswith("https://"), work["name"]+" documented date source/basis")
        matching = [row for row in featured_rows if re.sub(r"v\d+$","",re.search(r"\]\(([^)]+)\)",row).group(1)) == re.sub(r"v\d+$","",work["source"])]
        require(len(matching) == 1 and work["publicationDate"] in matching[0].split("|")[2], work["name"]+" README date matches catalog")
    agentx = next(w for w in featured if w["ref"] == 3)
    require(agentx["dateBasis"] == "official-announcement" and agentx["publicationDate"] == "2026-08-24", "AgentX official announcement date distinguished from paper dates")
    require("figures/09_evaluation_taxonomy.png" in readme and "| 分析事项 |" in readme, "README foregrounds workload analysis diagram and matrix")
    require("figures/11_workload_state_lifecycle.png" in readme, "README state lifecycle diagram")
    for axis in ("循环与阶段","上下文演化","缓存与重算","间隔与状态","整任务关键路径","控制面与工具冗余","资源与阶段耦合","并发与有效收益"):
        require(axis in readme and axis in report, axis+" workload question in README/report")
    require(readme.index("figures/09_") < readme.index("figures/01_"), "literature before experimental figure in README")
    for name in ("02_real_task_time", "03_calls_tokens", "04_notification_valleys", "07_resource_timelines"):
        require(f"figures/{name}.png" in readme, name+" prominently shown in README")
    task_names = {"task01":"前缀缓存与Prefill调度修复", "task02":"PP/offload槽宽与几何处理",
                  "task03":"GTCRN ConvTranspose量化修复", "task04":"ViT量化图冗余操作优化"}
    for task, name in task_names.items():
        cells = []
        for key in ("v41", "v4", "qwen"):
            case = next(c for c in real["cases"] if c["modelKey"] == key and c["task"] == task)
            state = ("自报完成" if case["status"] == "self-reported" else
                     ("完整验收通过" if task == "task03" else "已有产物验收通过") if case["status"] == "passed" else
                     "历史中止" if case["legacyStop"] else "限时未完成")
            cells.append(f'{case["elapsedMinutes"]:.2f}/{state}')
        require("| "+" | ".join(["任务"+str(int(task[-2:]))+"："+name, *cells])+" |" in readme, task+" README elapsed/status row")
    task3 = [next(c for c in real["cases"] if c["task"] == "task03" and c["modelKey"] == k) for k in ("v41","qwen")]
    for title, field in (("整题(min)","elapsedMinutes"),("工具不与API重叠(min)","toolMinutes")):
        require("| "+" | ".join([title,*[f'{c[field]:.2f}' for c in task3]])+" |" in readme, title+" README paired example")
    require("| 模型API(min/占整题) | "+" | ".join(f'{c["apiMinutes"]:.2f}/{100*c["apiMinutes"]/c["elapsedMinutes"]:.2f}%' for c in task3)+" |" in readme, "README task3 API time/share")
    for model in replay["models"].values():
        best = max((r for r in replay["runs"] if r["model"] == model and metrics(r)["sla_pass"]), key=lambda r:r["concurrency"])
        require(f'| {aliases[model]} | {best["concurrency"]} | {best["ttft"]["p90"]/1000:.3f} | {best["tpot"]["p90"]:.2f} | {metrics(best)["total_tpgs"]:.2f} |' in readme, model+" README measured SLA point")
    for label, count in (("上下文压缩",20),("system前缀变化",46),("输入增长、分母稀释",28)):
        require(f"| {label} | {count} |" in readme, label+" README annotation count")
    pdf_manifest = json.loads((ROOT/"paper/report_pdf.json").read_text(encoding="utf-8"))
    pdf_path = ROOT/"paper/report.zh-CN.pdf"
    require(pdf_path.read_bytes().startswith(b"%PDF-") and pdf_path.stat().st_size > 100000, "nontrivial PDF")
    require(pdf_manifest["sourceSha256"] == hashlib.sha256((ROOT/pdf_manifest["source"]).read_bytes()).hexdigest(), "PDF source Markdown synchronized")
    require(pdf_manifest["pdfSha256"] == hashlib.sha256(pdf_path.read_bytes()).hexdigest(), "PDF file fingerprint")
    require(pdf_manifest["pageCount"] > 10 and pdf_manifest["outlineCount"] >= 40 and all(pdf_manifest["checks"].values()), "PDF build content/layout checks recorded")
    require(len(pdf_manifest["embeddedFiguresSha256"]) == 11, "all 11 figures embedded in PDF")
    for name, expected in pdf_manifest["embeddedFiguresSha256"].items():
        require(expected == hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), name+" PDF figure synchronized")
    require(all(f"## {i}. {name}" in report for i,name in enumerate(["引言","相关工作","方法","实验","实验结果"],1)), "required paper sections")
    require("https://arxiv.org/pdf/2608.15127" in report and "https://papers.cool/arxiv/search?highlight=1&query=" in report, "both requested literature links")
    v41_c56 = next(r for r in replay["runs"] if r["model"] == replay["models"]["v41"] and r["concurrency"] == 56)
    require(f'{v41_c56["ttft"]["p99"]/1000:.3f} s' in report and f'{v41_c56["ttft"]["max"]/1000:.3f} s' in report, "C56 P99/Max warning")
    for key in ("v41","v4","qwen"):
        cs = [c for c in real["cases"] if c["modelKey"] == key]
        vals = [sum(c[field] for c in cs) for field in ("modelCalls","toolCalls","auxiliaryCalls","inputTokens","cachedTokens","outputTokens")]
        require("| " + " | ".join([{"v41":"V4.1","v4":"V4","qwen":"Qwen"}[key], *map(str,vals)]) + " |" in report, key+" report token totals")
        require("| " + " | ".join([{"v41":"V4.1","v4":"V4","qwen":"Qwen"}[key], *map(str,(vals[0],vals[1],vals[3],vals[5]))]) + " |" in readme, key+" README call/token totals")
    for doc in ROOT.rglob("*.md"):
        if any(part in EXCLUDED for part in doc.relative_to(ROOT).parts): continue
        text = doc.read_text(encoding="utf-8")
        require("**" not in text and "<br>" not in text, str(doc.relative_to(ROOT))+" plain formatting")
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if re.match(r"https?://|mailto:|#",target): continue
            resolved = (doc.parent/target.split("#")[0]).resolve()
            # This invocation creates its own verification artifact at the end.
            require(resolved.is_file() or resolved == ROOT/"verification.json", str(doc.relative_to(ROOT))+" link "+target)
    # Scope scan applies to published prose/data rather than this scanner's rules.
    deny = [r"/nfs/",r"/data/ssd",r"/home/",r"C:\\Users",r"(?i)bearer\s+[A-Za-z0-9]",r"(?i)(api[_-]?key|password|authorization)\s*[=:]\s*[\"']"]
    for directory in ("data","paper","docs","tables"):
        for path in (ROOT/directory).rglob("*"):
            if not path.is_file() or path.suffix not in (".md", ".json", ".csv", ".txt"): continue
            text = path.read_text(encoding="utf-8")
            require(not any(re.search(p,text) for p in deny), str(path.relative_to(ROOT))+" no private paths/credentials")
            for ip in re.findall(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])", text):
                try: address = ipaddress.ip_address(ip)
                except ValueError: continue
                require(not address.is_private, str(path.relative_to(ROOT))+" no private IPs")
    for directory in ("data",):
        for path in (ROOT/directory).glob("*.json"):
            # Original prompt fields and runtime topology must not be exported.
            text = path.read_text(encoding="utf-8")
            require(not re.search(r'"(?:messages|content|checkpoint|workspaceId|api_base|request_id|tool_use_id)"\s*:',text), path.name+" no raw message/deployment fields")
    pngs = sorted((ROOT/"figures").glob("*.png"))
    svgs = sorted((ROOT/"figures").glob("*.svg"))
    require(len(pngs) == len(svgs) == 11, "8 experimental and 3 survey PNG/SVG pairs")
    require({p.stem for p in pngs} == {p.stem for p in svgs}, "matching figure pairs")
    for path in pngs:
        require(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") and path.stat().st_size > 10000, path.name+" valid nontrivial PNG")
    fingerprints = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.rglob("*")) if p.is_file() and p.name != "verification.json" and not any(part in EXCLUDED for part in p.relative_to(ROOT).parts)}
    result = {"date":"2026-10-20", "passed":True, "checks":len(CHECKS), "replayRuns":15, "realCases":12, "formalRequests":2349, "coreWorkloadWorks":18,"scenarioBackgroundWorks":4,"figurePairs":11,"pdfPages":pdf_manifest["pageCount"], "independentRawEvidenceAudit":False, "scope":"Released numeric evidence, workload review structure and report/PDF consistency only; not an independent source audit", "filesSha256":fingerprints}
    (ROOT/"verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "filesSha256"},ensure_ascii=False))


if __name__ == "__main__":
    main()
