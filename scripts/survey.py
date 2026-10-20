#!/usr/bin/env python3
"""Render a focused, source-linked literature index and original SVG diagrams."""
import html
import json
from pathlib import Path
from report_browser import ROOT, ReportBrowser

FIG = ROOT / "figures"
FONT = '"WenQuanYi Micro Hei", "Noto Sans CJK SC", sans-serif'
RED, ROSE, CORAL, LILAC, INK = "#AF2E3D", "#CB5979", "#E97F68", "#BDA9C4", "#263040"


class Diagram:
    def __init__(self, width, height, title, subtitle):
        self.width,self.height = width,height
        self.items = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10" fill="#7b8492"/></marker></defs>']
        self.text(width/2,53,[title],32,RED)
        self.text(width/2,91,[subtitle],19,"#666d79")

    def text(self,x,y,lines,size=22,color=INK,anchor="middle"):
        for i,line in enumerate(lines):
            self.items.append(f'<text x="{x}" y="{y+i*(size+11)}" text-anchor="{anchor}" font-family="{html.escape(FONT,quote=True)}" font-size="{size}" fill="{color}">{html.escape(str(line))}</text>')

    def box(self,x,y,w,h,lines,stroke=ROSE,fill="#fff7f8",size=22,dashed=False):
        dash=' stroke-dasharray="8 6"' if dashed else ''
        self.items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="2"{dash}/>')
        self.text(x+w/2,y+h/2-(len(lines)-1)*(size+11)/2+size*.32,lines,size)

    def arrow(self,points,dashed=False):
        pts=" ".join(f"{x},{y}" for x,y in points)
        dash=' stroke-dasharray="8 6"' if dashed else ''
        self.items.append(f'<polyline points="{pts}" fill="none" stroke="#7b8492" stroke-width="2" marker-end="url(#arrow)"{dash}/>')

    def save(self,name):
        target=FIG/(name+".svg")
        target.write_text("\n".join(self.items+["</svg>"])+"\n",encoding="utf-8")
        return target


def taxonomy():
    d=Diagram(1560,1100,"Coding / Work Agent：多轮 Workload 分析框架","关注负载如何产生服务压力，而非 Agent 能力或题库分类。方法归纳，非新增实测。")
    cards=[
        ("1  循环与模型重入",["用户需求 / 模型 step / 工具动作","自主轮数 / 重试 / 分支 / 阶段","同一并发 ≠ 同一请求到达"],RED),
        ("2  上下文演化",["历史 / 新工具结果 / 输出","增长、压缩与前缀改变","长 ISL ≠ 大量实际 prefill"],CORAL),
        ("3  缓存与重复计算",["命中量 / 待 prefill / 真正新增","轮内与跨轮、暂停与模型切换","不匹配、过期和驱逐要区分"],ROSE),
        ("4  间隔与状态寿命",["工具等待 / 用户批准 / 长暂停","KV、文件、进程与业务状态","驻留 GB·s / 恢复首轮延迟"],LILAC),
        ("5  整任务关键路径",["API / 工具 / 编排 / 队列","并行与后台执行不能简单累加","请求更快不保证整题更快"],RED),
        ("6  控制面与工具冗余",["工具定义、观察、摘要和重试","辅助调用 / token / 数据移动","安全复用先看权限与新鲜度"],CORAL),
        ("7  跨层资源耦合",["GPU / CPU / DRAM / I/O / 网络","阶段变化、突发和共享干扰","利用率相关性不代替干预"],ROSE),
        ("8  并发与有效收益",["活跃请求 ≠ 等待 session","到达、SLA、质量与公平性","完成时间 / goodput / 资源"],LILAC),
    ]
    for i,(title,lines,color) in enumerate(cards):
        x=25+(i%4)*385;y=160+(i//4)*330
        d.box(x,y,360,280,[],color,"#fcf9fa")
        d.text(x+180,y+48,[title],23,color)
        d.text(x+180,y+115,lines,18)
    d.arrow([(780,785),(780,827)])
    d.box(40,840,1480,155,[],RED,"#f7f3f6")
    d.text(780,883,["质量 / SLA 约束下的整任务时间、有效吞吐与资源占用"],25,RED)
    d.text(780,937,["请求指标 → 跨轮状态 → 关键路径 → 受控策略验证"],23)
    d.text(780,1048,["多轮 Workload 不只是把聊天轮数加长；依赖、间隔、状态和控制面都会改变服务压力。"],21,"#646b78")
    return d.save("09_evaluation_taxonomy")


def paradigms():
    d=Diagram(1560,1020,"多轮 Workload：三种互补测量方法","依据 AgentX、TraceLab、AgentSysBench 与 AgentSight 归纳；任务闭环测执行路径，不是能力榜。")
    for x,title,color in [(25,"A  固定轨迹回放",RED),(540,"B  任务自主闭环",CORAL),(1055,"C  全栈关联与生产补充",ROSE)]:
        d.box(x,135,480,800,[],color,"#fffdfd")
        d.text(x+240,181,[title],25,color)
    for x,items in [(65,[("固定请求、长度、前缀结构",),("并发 / 到达计划",),("推理服务与 KV 缓存",),("延迟 / 吞吐 / 缓存",)]),(1095,[("任务、模型、工具事件",),("状态与 CPU / GPU / I/O",),("身份、时间与依赖关联",),("关键路径 / 状态寿命 / 质量",)])]:
        for i,lines in enumerate(items):
            y=235+i*135
            d.box(x,y,400,85,lines,RED if x==65 else ROSE,size=22)
            if i<3:d.arrow([(x+200,y+88),(x+200,y+129)])
    d.box(65,815,400,83,["长度回放不重建工具决策"],"#999fa9","#fafafa",20,True)
    bx=590
    for y,lines in [(235,["任务与可恢复的初始环境"]),(360,["Agent + 模型 + 执行接口"]),(485,["代码 / 浏览器 / 业务工具"]),(610,["观察、日志与环境状态"])]:
        d.box(bx,y,380,80,lines,CORAL,size=21)
        if y<610:d.arrow([(780,y+83),(780,y+119)])
    d.arrow([(590,650),(562,650),(562,400),(587,400)])
    d.text(570,535,["循环"],17,"#777",anchor="end")
    d.arrow([(780,695),(780,741)])
    d.box(590,748,380,85,["结束 / 停止 → 独立验收"],CORAL,size=21)
    d.text(780,885,["路径可变：测时间、状态与开销"],20,"#646b78")
    d.box(1095,815,400,83,["观测相关性 ≠ 优化因果证据"],"#999fa9","#fafafa",20,True)
    d.text(780,983,["先用回放隔离引擎因素，再用任务闭环看端到端收益；生产等待与长期状态需单独补充。"],21,RED)
    return d.save("10_evaluation_paradigms")


def architectures():
    d=Diagram(1560,1100,"多轮 Workload：等待期间的状态与跨层信息","依据 AgentSysBench、InferCept、Continuum、CacheWise 与 HEAR 归纳；不是已部署的本地优化。")
    labels=[(40,["运行","模型 / 工具 / 编排"],RED),(430,["等待","工具 / 用户 / 外部事件"],CORAL),(820,["恢复","下一轮模型进入"],ROSE),(1210,["结束 / 交接","释放或持久化"],LILAC)]
    for x,lines,color in labels:
        d.box(x,205,310,105,lines,color,size=23)
    for a,b in [(353,427),(743,817),(1133,1207)]: d.arrow([(a,257),(b,257)])
    d.arrow([(975,202),(975,146),(195,146),(195,202)])
    d.text(590,134,["未完成：继续下一轮"],19,"#646b78")
    d.text(780,365,["等待不是结束：分开管理三类状态 / 资源"],26,RED)
    blocks=[
        (40,"性能状态（可重算）",["KV：HBM 驻留 / DRAM 卸载 / 驱逐","比较复用、重算、传输与排队代价","观察恢复首轮和内存时间积分"],RED),
        (550,"正确性状态（保持语义）",["文件 / 进程状态 / 数据库 / 业务副作用","暂停或迁移仍需一致、可恢复","不能当作普通 KV 随意丢弃"],CORAL),
        (1060,"活动工作集（临时资源）",["编译、测试的 CPU / DRAM 峰值","运行峰值 ≠ 全部检查点大小","暂停边界与实际驻留需测量"],ROSE),
    ]
    for x,title,lines,color in blocks:
        d.box(x,405,460,225,[],color,"#fcf9fa")
        d.text(x+230,452,[title],23,color)
        d.text(x+230,503,lines,19)
    d.box(40,705,700,145,["Harness 知道：步骤依赖、工具事件、恢复意图","记录上下文变化、用户间隔和任务完成目标"],RED,"#fff7f8",21)
    d.box(820,705,700,145,["Engine 知道：队列、KV、prefill / decode","记录驱逐、换入、内存压力与执行能力"],ROSE,"#fff7f8",21)
    d.arrow([(744,761),(816,761)])
    d.arrow([(816,793),(744,793)])
    d.text(780,680,["跨层关联"],18,"#646b78")
    d.arrow([(390,853),(390,886),(780,886),(780,909)])
    d.arrow([(1170,853),(1170,886),(780,886)])
    d.box(40,922,1480,110,["用 session / step / parent span 与对齐时钟关联，而不是只叠 GPU 利用率曲线","测量占用、等待、恢复与关键路径；改变策略后，再验证质量和端到端收益"],RED,"#f7f3f6",22)
    return d.save("11_workload_state_lifecycle")


def index_and_bib(data):
    works={w["ref"]:w for w in data["works"]}
    assigned=[r for g in data["focusGroups"] for r in g["refs"]]
    assert len(assigned)==len(set(assigned))==len(works)==data["coreWorks"]+data["backgroundWorks"]
    assert set(assigned)==set(works)
    assert sum(w["scope"]=="core" for w in works.values())==data["coreWorks"]
    lines=["# Coding / Work场景的多轮Workload研究索引", "", f'核对日期：{data["reviewDate"]}。核心{data["coreWorks"]}项研究/工具/硬件方法，场景背景{data["backgroundWorks"]}项；不是Agent框架、题库或能力综述。', "", "## 选取与核验原则", "", "优先原论文与作者材料，纳入直接分析多轮轨迹、上下文/KV、工具等待、任务调度、跨层资源或观测的方法，并补充P/D交接与decode硬件评估。仅凭名称包含Agent不构成纳入理由；普通聊天只作对照。不是穷尽式系统综述，不混排论文性能，不把上游收益当成本地复现。", "", "方法与测量对象来自原工作，边界是本文用于避免误读的归纳。版本化链接固定本次所引稿件；年份取首次发布或所引会议版本。图R1～R3为负载与测量方法归纳，不复制原图。DCP是官方工程材料，HeteroPanacea是模拟研究，LLMCompass是通用硬件方法，不能统一称为Agent轨迹论文。"]
    for group in data["focusGroups"]:
        lines += ["", "## "+group["name"], "", "| 研究/工具 | 考察的负载与方法 | 测量什么 | 适用边界 |", "|---|---|---|---|"]
        for ref in group["refs"]:
            w=works[ref]
            version="，"+w["version"] if w.get("version") else ""
            lines.append(f'| [{w["name"]}]({w["source"]}) [{ref}]{version} | {w["method"]} | {w["measure"]} | {w["boundary"]} |')
    lines += ["", "## 使用说明", "", "前四组为Workload/系统主线，第五组补充P/D交接与decode硬件方法；最后一组只解释Coding与企业工作的场景来源，不展开能力评测。AgentX是回放工具，papers.cool是AgentSysBench论文的检索入口，不重复计为另一篇研究。", "", "本报告统一术语：session可含多个用户需求，每个需求含多次模型API和工具动作。不同论文的request/turn/step含义必须先对齐再比较。", "", "报告第2.10节对应文献机制与设计问题，第3.6节给出E1～E4待执行方案；新文献未作为本地已复现性能或ASIC测量。"]
    (ROOT/"docs/literature-index.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    entries=[]
    for w in sorted(works.values(),key=lambda w:w["ref"]):
        if w["ref"]<8: continue
        entries.append(f'@misc{{surveyRef{w["ref"]},\n  title={{{w["title"]}}},\n  author={{{w["authors"]}}},\n  year={{{w["year"]}}},\n  url={{{w["source"]}}},\n  note={{Primary source checked 2026-10-20; workload/serving methods review, not reproduced performance}}\n}}')
    (ROOT/"references.survey.bib").write_text("\n\n".join(entries)+"\n",encoding="utf-8")


def main():
    data=json.loads((ROOT/"data/literature.json").read_text())
    index_and_bib(data)
    assets=[taxonomy(),paradigms(),architectures()]
    with ReportBrowser() as browser:
        for asset in assets:
            dims={"09_evaluation_taxonomy":(1560,1100),"10_evaluation_paradigms":(1560,1020),"11_workload_state_lifecycle":(1560,1100)}[asset.stem]
            browser.screenshot(asset,asset.with_suffix(".png"),*dims)
    print(json.dumps({"coreWorks":data["coreWorks"],"backgroundWorks":data["backgroundWorks"],"diagrams":len(assets),"formats":["svg","png"]}))


if __name__=="__main__": main()
