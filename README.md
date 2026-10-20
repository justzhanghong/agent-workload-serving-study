# Agent Workload Serving Study

从多轮回放到真实 Agent 任务：三模型推理服务性能与前缀缓存行为研究。

模型：DeepSeek-V4.1-Flash、DeepSeek-V4-Flash-0731、Qwen3.8-27B。

## 阅读报告

[技术报告全文（中文）](paper/report.zh-CN.md)：摘要、引言、相关工作、方法、实验、实验结果、讨论与局限、结论、参考文献。

报告结合 15 个 AgentX 并发测点、12 组真实代码任务、2349 次正式模型请求，以及 CUDA Graph 和动态通知专项分析。只整理已完成实验，不新增推理测试。

1. 当前 AgentX 子集与 P90 SLA 下，V4.1 的最高已测可行并发为 56，V4 与 Qwen 为 48；不是生产最大容量。
2. 真实任务主要时间落在工具与等待区间；回放推理性能不能直接预测整题耗时。
3. 已验证的缓存问题包括动态 system 通知前置；没有把全部缓存波谷归为显存不足，也未测修复后的加速收益。

![AgentX 吞吐—延迟](figures/01_agentx_pareto.png)

## 数据与复现

- [数据说明](data/README.md)、[证据指纹](data/provenance.json)
- [AgentX 汇总](tables/agentx_summary.md)、[完整分位数](tables/agentx_percentiles.md)、[有效数据集](tables/dataset.md)
- [真实任务耗时与占比](tables/real_tasks.md)、[调用与 Token](tables/calls_tokens.md)
- [环境与版本](docs/environment.md)、[消融补充](docs/ablations.md)、[复现范围](docs/reproducibility.md)

```bash
python3 scripts/analyze.py
python3 -m pip install -r requirements.txt
python3 scripts/plot.py
python3 scripts/verify.py
```

数值分析与校验仅使用 Python 标准库；绘图使用 Matplotlib，均不需要 GPU 或模型 API。上述命令在仓库根目录执行。[校验摘要](verification.json)记录本次交付检查。

原始提示词、题目源码、工具输入输出、账号及部署地址不在仓库内。只提供脱敏测量数据和图表；不能据此独立重跑未公开任务或验证隐藏答案。第三方论文、软件、数据与任务材料的权利归原权利人，不随本仓库重新授权。
