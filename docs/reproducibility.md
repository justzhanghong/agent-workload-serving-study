# 复现与审计

## 公开材料能复现什么

在 Python 3.10 或以上环境，从仓库根目录执行：

```bash
python3 scripts/analyze.py
python3 -m pip install -r requirements.txt
python3 scripts/plot.py
python3 scripts/verify.py
```

分析和校验只依赖标准库，绘图依赖 Matplotlib。无需 GPU、模型权重、API、Windows 或内部网络；Windows 跳板仅用于原交付流程的 Git 推送，不属于图表复现依赖。

可重新计算吞吐、SLA判断、加权缓存率、真实任务区间占比、配对 Graph 差值、缓存事件规则及历史三轮差值；可生成实验汇总表和前八组PNG/SVG。PNG可能受字体和Matplotlib平台差异影响，不承诺跨平台逐字节一致。SVG设置固定日期，但也不以其字节相同代替数值正确性。

AgentX只公开各运行的分布摘要，不能从摘要重新计算原始请求分位数；校验脚本检查它们的顺序和本文引用值。真实任务只公开数值请求与采样，不含原始消息、代码与工具参数，因此不能独立复验通知文本语义、题目正确性或完整数据采集链路。

## 调研图与PDF重建

额外安装Chrome/Chromium和中文字体（本次使用WenQuanYi Micro Hei），然后执行：

```bash
python3 -m pip install -r requirements-docs.txt
python3 scripts/survey.py
python3 scripts/build_pdf.py
python3 scripts/verify.py
```

可用`AGENT_STUDY_CHROME`指定浏览器可执行文件。脚本创建自己的临时浏览器配置，不使用用户日常浏览器配置，也不调用模型。`survey.py`依据`data/literature.json`生成文献索引、补充BibTeX及三组原生SVG/PNG方法图。它不自动更新论文或重新执行被综述的任务。

PDF的唯一正文来源为`paper/report.zh-CN.md`，使用同名SVG嵌入报告图，添加章节目录、书签与页码。`paper/report_pdf.json`保存正文、图像和PDF的SHA256、页数及构建检查。修改正文或图像后必须重建PDF；`verify.py`检查指纹一致性。HTML与预览页位于忽略的临时构建目录，不作为另一份手写正文发布。浏览器/字体差异可能改变分页，指纹证明本次文件对应，不承诺跨平台PDF字节完全相同。

## Decode芯片方案的复现边界

[报告第2.10节与第3.6节](../paper/report.zh-CN.md)给出文献机制和E1～E4实验设计。E1已增加独立稀疏attention计时/计数器，以及整模型TP4理论对齐、核心频率敏感性和独立profile；它们不等于完成全部E1，更不等于完成E2状态恢复、E3等预算PD或E4芯片设计空间。没有提供目标ASIC规格或ASIC实测结果。

已发布新数据可离线重建：

```bash
python3 scripts/plot_decode_microbench.py
python3 scripts/plot_decode_theory.py
python3 scripts/summarize_decode_theory.py
python3 scripts/test_decode_probes.py
python3 scripts/test_decode_theory.py
python3 scripts/build_pdf.py
python3 scripts/verify.py
```

TP4实验使用固定到main具体提交的部分理论，3次全新服务共48个测点，其中42个正常计时、6个独立profiler；另用1个新服务内3个block补18个固定输出token测点，不混作3次独立服务。24份逐rank trace汇总、逐组输出一致性检查与原始指纹见`data/decode_theory_tp4.json`。HBM调频命令未生效，6个相应测点保留为无效干预，不能计算HBM弹性。[详细协议](decode-theory-tp4-protocol.md)说明实际频率核验、匹配batch和重叠处理。

原材料所有者可用`analyze_decode_theory.py --root <原始实验目录> --output <脱敏输出JSON>`重新导出；默认拒绝未完成实验，`--allow-partial`仅供运行中诊断，不用于发布。`analyze_decode_microbench.py`导出独立单算子与计数器数据。原始日志/trace仍非公开材料；脚本和指纹不能取代第三方原始证据审计。

后续实施需先确认模型/注意力架构、精度、prefill芯片接口和可用内存/互连约束，补充细粒度profiling与状态恢复记录，再校准或扩展模拟器。LLMCompass与HeteroPanacea仅为方法参考，不声称本仓库已集成这些工具或复现论文收益。校验脚本检查新文献和计划状态的一致性，不替代外部论文核验、硬件预测验证或任务质量验收。

## 图表索引

| 文件前缀 | 内容 | 关键边界 |
|---|---|---|
| 01_agentx_pareto | TPGS—P90 interactivity/TTFT 双图 | 逻辑总吞吐含缓存输入；Qwen有效集合不同 |
| 02_real_task_time | 四题耗时与互斥分解 | 自报、通过、停止不混同；工具总并集另外列出 |
| 03_calls_tokens | 调用次数与累计 token | 不同执行路径/结束状态，不是费用排名 |
| 04_notification_valleys | 三个服务的通知波谷 | 当前安装源码已核验，不泛指所有版本 |
| 05_graph_ablation | Graph配对均值 | 不代表逐请求固定收益 |
| 06_cache_condition | 历史缓存条件与三轮范围 | 范围不是置信区间；内部输出不一致 |
| 07_resource_timelines | GPU/KV时序 | GPU每分钟样本均值、KV每分钟峰值，无缺失插值 |
| 08_latest_client_control | 新版客户端实际请求对照 | 独立服务链路的行为检查，不是性能基准 |
| 09_evaluation_taxonomy | Coding / Work多轮负载的八项分析轴 | 文献方法归纳，不是Agent能力分类或新增实验 |
| 10_evaluation_paradigms | 固定回放、自主执行、全栈关联 | 决策闭环与负载闭环不是同一变量 |
| 11_workload_state_lifecycle | 运行/等待/恢复、三类状态与harness/engine信息 | 分析示意，不声称已在本地接入卸载或协同策略 |
| 12_decode_sparse_microbench | 独立稀疏attention接口时间与主kernel DRAM读取 | 不是Mega或整模型；两个采集协议不混为同一计时 |
| 13_decode_theory_tp4 | 部分理论、四卡整模型核心频率敏感性及独立profile | HBM干预未生效；核心敏感不等于Tensor Core瓶颈 |

## 所有者从原材料重新导出

`scripts/export_evidence.py --help` 列出输入。导出器只读取既有摘要、配对CSV及历史绘图脚本中的字面量数据，不执行旧绘图脚本，不启动模型。使用显式字段白名单，丢弃权重路径、部署地址、原始会话ID及来源路径。

九个输入文件分别是回放摘要、真实任务数值摘要、Graph汇总、零命中三轮汇总、新客户端通知对照、历史正常缓存绘图源、合成网格汇总、公开Graph配对CSV、内部Graph配对CSV。它们的完整文件SHA256记录于 [provenance.json](../data/provenance.json)；原始真实任务证据另保留去路径的内容指纹集合。这些指纹由所有者与受控归档核对，不是公开下载链接。

## 校验与证据强度

校验脚本检查15点的固定有效样本和上下文规则、2349请求对账、状态数量、区间分解、辅助调用及事件计数、Graph逐数值对的汇总、历史三轮差值、通知行为对照、单算子计时/计数器、TP4测点及实际频率、profile区间并集、正文关键表格、文献分组与引用、内部链接、13组图像、PDF同源指纹和发布内容脱敏。

该校验验证“仓库数值与报告之间的一致性”，不是对私有原始采集的第三方独立认证。原始验收、客户端消息语义和软件安装身份的审计证据由原实验所有者保管；正文准确注明其范围和缺口。
