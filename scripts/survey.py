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
    d=Diagram(1560,1040,"Coding / Work Agent：评测对象与方法体系","任务主线优先；服务、观测与可信验收横向支撑。非论文原图，非新增实验。")
    cards=[
        ("Coding 与终端工程",["仓库修复 / 多文件修改 / 运行环境", "SWE-bench / Pro / Live", "Terminal-Bench / SWE-agent / OpenHands"],RED),
        ("Work 与企业工作流",["企业应用 / 浏览器 / 桌面 / 文件", "WorkArena / TheAgentCompany / CRM", "OSWorld / SpreadsheetBench / 浏览检索"],CORAL),
        ("科研与 ML 工程",["编写程序 / 训练 / 实验 / 复现", "MLE-bench / ScienceAgentBench", "PaperBench：分层检查复现产物"],ROSE),
        ("服务与可观测性",["固定轨迹 / 推理服务 / 全栈测量", "AgentX / AgentSysBench / AgentSight", "状态管理 / 依赖 / CPU、GPU、I/O"],LILAC),
        ("评测方法与可靠性",["过程进展 / 重复运行 / 独立验收", "规则、安全、基础设施与失败分类", "多角色依赖和任务关键路径"],ROSE),
        ("普通对话与组件：简要背景",["聊天记忆、函数调用等只作补充", "不能代替代码、工件与业务终态验收", "本综述不展开普通聊天能力排名"],"#a3a8b1"),
    ]
    for i,(title,lines,color) in enumerate(cards):
        x=40+(i%3)*510;y=150+(i//3)*315
        d.box(x,y,480,265,[],color,"#fcf9fa")
        d.text(x+240,y+45,[title],25,color)
        d.text(x+240,y+105,lines,20)
    d.arrow([(780,750),(780,790)])
    d.box(40,805,1480,150,[],RED,"#f7f3f6")
    d.text(780,843,["统一比较维度"],24,RED)
    d.text(780,889,["评测目标  /  任务与模态  /  执行驱动  /  状态范围  /  到达与资源  /  判定与重复"],22)
    d.text(780,990,["读图结论：工作是否做对、服务是否够快、系统为何变慢，是互补问题，不能合成一个未经定义的排名。"],20,"#646b78")
    return d.save("09_evaluation_taxonomy")


def paradigms():
    d=Diagram(1560,1020,"三种互补实验范式：同一工作，三个观察切面","抽象自 AgentX、Coding / Work 任务基准及 AgentSysBench / AgentSight；虚线为可选或范围外。")
    for x,title,color in [(25,"A  固定轨迹回放",RED),(540,"B  任务自主闭环",CORAL),(1055,"C  全栈关联与生产补充",ROSE)]:
        d.box(x,135,480,800,[],color,"#fffdfd")
        d.text(x+240,181,[title],25,color)
    for x,items in [(65,[("固定请求、长度、前缀结构",),("并发 / 到达计划",),("推理服务与 KV 缓存",),("延迟 / 吞吐 / 缓存",)]),(1095,[("任务、模型、工具事件",),("状态与 CPU / GPU / I/O",),("身份、时间与依赖关联",),("关键路径 / 状态寿命 / 质量",)])]:
        for i,lines in enumerate(items):
            y=235+i*135
            d.box(x,y,400,85,lines,RED if x==65 else ROSE,size=22)
            if i<3:d.arrow([(x+200,y+88),(x+200,y+129)])
    d.box(65,815,400,83,["不验证新模型是否解对任务"],"#999fa9","#fafafa",20,True)
    bx=590
    for y,lines in [(235,["任务与可恢复的初始环境"]),(360,["Agent + 模型 + 执行接口"]),(485,["代码 / 浏览器 / 业务工具"]),(610,["观察、日志与环境状态"])]:
        d.box(bx,y,380,80,lines,CORAL,size=21)
        if y<610:d.arrow([(780,y+83),(780,y+119)])
    d.arrow([(590,650),(562,650),(562,400),(587,400)])
    d.text(570,535,["循环"],17,"#777",anchor="end")
    d.arrow([(780,695),(780,741)])
    d.box(590,748,380,85,["结束 / 停止 → 独立验收"],CORAL,size=21)
    d.text(780,885,["路径可变：比较真实工作结果"],20,"#646b78")
    d.box(1095,815,400,83,["观测相关性 ≠ 优化因果证据"],"#999fa9","#fafafa",20,True)
    d.text(780,983,["先用回放隔离引擎因素，再用任务闭环看端到端收益；生产等待与长期状态需单独补充。"],21,RED)
    return d.save("10_evaluation_paradigms")


def architectures():
    d=Diagram(1560,1070,"Coding / Work Agent：执行器与评测边界","方法示意，依据 SWE-agent [32]、Agentless [35]、OpenHands / Versa [36,47] 重绘；不是原项目完整架构。")
    for x,title,color in [(25,"SWE-agent：ACI 交互闭环",RED),(540,"Agentless：固定阶段",CORAL),(1055,"OpenHands / Versa：通用工具",ROSE)]:
        d.box(x,135,480,735,[],color,"#fffdfd")
        d.text(x+240,181,[title],23,color)
    d.box(70,235,390,85,["模型根据观察选择下一步"],RED)
    d.arrow([(265,323),(265,365)])
    d.box(70,372,390,85,["Agent-Computer Interface"],RED)
    d.arrow([(265,460),(265,500)])
    d.box(70,508,390,110,["仓库搜索 / 阅读 / 编辑", "命令执行 / 测试"],RED)
    d.arrow([(265,621),(265,663)])
    d.box(70,671,390,85,["工具结果与新观察"],RED)
    d.arrow([(70,713),(47,713),(47,278),(67,278)])
    for i,lines in enumerate([["定位相关文件与代码"],["生成修复候选"],["验证与筛选补丁"]]):
        y=260+i*170
        d.box(585,y,390,100,lines,CORAL)
        if i<2:d.arrow([(780,y+103),(780,y+160)])
    d.text(780,804,["阶段固定，可比较多候选与验证预算"],20,"#646b78")
    d.box(1100,235,390,100,["模型与事件 / 执行编排"],ROSE)
    d.arrow([(1295,338),(1295,383)])
    for y,lines in [(390,["代码编辑 / 执行 / 终端"]),(510,["搜索 / 视觉网页浏览"]),(630,["文件访问 / 沙盒 / 工件"])]:
        d.box(1100,y,390,85,lines,ROSE,size=22)
    d.arrow([(1493,673),(1515,673),(1515,285),(1493,285)])
    d.text(1295,804,["同一工具平台连接 Coding 与 Work"],20,"#646b78")
    d.arrow([(265,873),(265,911)],True)
    d.arrow([(780,873),(780,911)],True)
    d.arrow([(1295,873),(1295,911)],True)
    d.box(40,920,1480,100,["评测边界：固定题目、初态与预算 → 保存候选 / 轨迹 → 测试、工件或业务终态验收", "分开执行器效果、模型效果与环境故障；自报完成不是独立验收。"],RED,"#f7f3f6",21,True)
    return d.save("11_coding_work_architectures")


def index_and_bib(data):
    works={w["ref"]:w for w in data["works"]}
    assigned=[r for g in data["focusGroups"] for r in g["refs"]]
    assert len(assigned)==len(set(assigned))==len(works)==43
    assert set(assigned)==set(works)
    lines=["# Coding / Work Agent文献方法索引", "", "核对日期：2026-10-20。正文重点是执行真实工作、可验证产物和服务系统；普通聊天/记忆仅作背景。", "", "## 选取与核验原则", "", "优先原论文、会议页面及作者项目；选择任务基准、执行框架、验收有效性和服务观测中方法不同的代表。不是穷尽性系统综述，不合并不同任务、版本、执行器、预算或验收方式的模型成绩，也不声称已复现这些基准。", "", "表中方法概述来自原工作，边界是本文用于避免误读的归纳。图R1～R3是方法重绘，不复制论文原图；已有实验与调研分开。"]
    for group in data["focusGroups"]:
        lines += ["", "## "+group["name"], "", "| 工作 | 执行/构造方法 | 检查什么 | 使用边界 |", "|---|---|---|---|"]
        for ref in group["refs"]:
            w=works[ref]
            lines.append(f'| [{w["name"]}]({w["source"]}) [{ref}] | {w["method"]} | {w["measure"]} | {w["boundary"]} |')
    lines += ["", "## 补充来源", "", "BFCL的多轮与多步方法以[作者V3说明](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html)补充会议论文；不把当前总榜构成套用到旧版。AgentX是上游回放工具，papers.cool是检索入口，都不伪称为独立基准论文。"]
    (ROOT/"docs/literature-index.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    entries=[]
    for w in sorted(works.values(),key=lambda w:w["ref"]):
        if w["ref"]<8: continue
        entries.append(f'@misc{{surveyRef{w["ref"]},\n  title={{{w["title"]}}},\n  author={{{w["authors"]}}},\n  year={{{w["year"]}}},\n  url={{{w["source"]}}},\n  note={{Primary source checked 2026-10-20; methods review, not a reproduced score}}\n}}')
    (ROOT/"references.survey.bib").write_text("\n\n".join(entries)+"\n",encoding="utf-8")


def main():
    data=json.loads((ROOT/"data/literature.json").read_text())
    index_and_bib(data)
    assets=[taxonomy(),paradigms(),architectures()]
    with ReportBrowser() as browser:
        for asset in assets:
            dims={"09_evaluation_taxonomy":(1560,1040),"10_evaluation_paradigms":(1560,1020),"11_coding_work_architectures":(1560,1070)}[asset.stem]
            browser.screenshot(asset,asset.with_suffix(".png"),*dims)
    print(json.dumps({"reviewWorks":len(data["works"]),"diagrams":len(assets),"formats":["svg","png"]}))


if __name__=="__main__": main()
