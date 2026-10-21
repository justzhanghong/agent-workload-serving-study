# 数据与可复现范围

本目录保存已完成实验的数值、事件类别和内容指纹，以及单独的公开文献目录；不包含原始提示词、工具输入输出、题面、提交源码、模型权重、账号或部署地址。

| 文件 | 内容 |
|---|---|
| agentx.json | 三模型、五个并发点的请求级分布摘要与运行总量；只含正常前缀缓存 |
| real_tasks.json | 十二组真实任务、2349 次请求的数值指标及资源序列 |
| cuda_graph_ablation.json | V4.1 CUDA Graph 两配置的严格配对统计，不含内部样本内容 |
| graph_paired.json | Graph 配对的纯数值记录，不含会话 ID、题面或消息内容 |
| micro_grid.json | C1 合成网格结果，区分三次中位数与后端消融的稳态样本 |
| cache_stability.json | 早期 C4 零缓存三轮复测；不是后期 2048 条并发实验 |
| cache_baseline.json | 正常缓存历史图的数值，不具备重复分布 |
| notification_control.json | 最新客户端通知对照的六次请求数值，不含请求正文 |
| provenance.json | 九个输入文件的 SHA256，以及受控原始证据的内容指纹 |
| literature.json | 23项多轮Workload/系统研究、工具与硬件方法，4项场景背景；不是实验结果数据 |
| decode_attention_microbench.json | 独立B200稀疏attention的108个形状测点、24个另采NCU样本；不代表Mega或整模型 |
| decode_theory_tp4.json | 3次服务的48个TP4测点与24份rank profile汇总，另1服务内18个固定token控制；无有效HBM调频 |

AgentX 中 `ttft`、`tpot` 的单位均为 ms，`isl`、`osl` 为 token，`cacheHit` 为请求级百分比，`cacheHitPercent` 为全局 token 加权百分比。吞吐由运行总量与完整回放时间计算，不是请求分位数。分位数使用线性插值，因此 token 分位数原始值可以为小数；展示表四舍五入为整数，不改变原数据。

真实任务中时间以分钟表示；请求 `ttftSeconds` 以秒表示，`tpotMs` 以毫秒表示。API 区间和工具总区间各自取并集，但彼此可以重叠；`toolMinutes` 仅保留不与 API 重叠的工具区间。压缩和通知事件以 1 起始请求序号定位。`resources.gpu` 为四卡平均利用率，`kv` 为分配 KV 块占用百分比，`cpuIo` 为可归属容器子集的 CPU 百分比、累计读 GB、累计写 GB；不能据此认定 HBM 带宽瓶颈。

三模型使用同一批 2048 条候选请求。DeepSeek 各执行 2048 条；Qwen 超上下文跳过 15 条，执行 2033 条。不同并发的同一模型有效集合和输出预算一致，跨模型不是共同有效样本上的严格等量比较。

`evidence.requestsSha256`、`evidence.manifestSha256` 是原始请求指标文件和清单的内容指纹，不是公开文件下载地址。原始日志由实验所有者受控保管，仓库只能复现数值分析和图表，不能从长度合成输入恢复真实代码语义，也不能独立复验未公开题目的正确性。

文献目录研究Coding / Work场景下的多轮负载，不是Agent能力或题库综述。`scope`区分核心研究与场景背景，`focusGroups`按轨迹、KV状态、调度接口和跨层资源分组。目录生成索引及书目，不修改九项原实验输入或证据指纹。检索日、原稿版本及方法边界随目录保存，不能把文献中的优化收益当成本地复现实测。

新增TP4的`stepMs`是实际匹配generation batch的后端迭代elapsed_ms，不是纯GPU时间或客户端TPOT。`summaries`对3次独立服务的步骤均值再汇总；`fixedTokenCases`则来自1个额外服务内3个block，不能当作3次独立服务。正常生成输出序列不同的事实保存在`frequencyComparisons`，受限输出的实际序列一致性另存`fixedTokenComparisons`。

`requestedMemoryMHz=3402`不代表调频生效：本轮相应测点`memoryInterventionValid=false`，实际仍3996 MHz。profile的kernel duration之和可能重叠，活动时间按区间并集计算，不作墙钟占比分解。理论字段为固定main提交、未校准且不完整的已计需求，不是硬件利用率或完整延迟预测。各项原始指纹只证明材料对应关系，不公开原始trace、token或部署拓扑。
