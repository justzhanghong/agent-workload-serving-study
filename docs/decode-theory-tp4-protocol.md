# V4.1四卡资源配比假说：验证协议

平台限定NVIDIA B200×4，TP4/EP4；本实验不使用P/D分离，也不模拟目标ASIC。

## 假说来源与边界

来源为[model_analysis仓库](https://github.com/Romaosir/model_analysis)的main提交`f2b5d167a75c2bb6a4b9385d2fd2de30f6b5278a`，其中`data/reports/3ee4c6e6-9a91-47c6-bfdb-6acd5e47edfe/hardware.md`与`data/module-skills/modelmeter-analysis-v2/scripts/estimate_deepseek_comparison.py`给出部分GPU资源需求。服务器工作区另有未提交的BS8公式，本轮不混用。

把假说拆成三个可分别判断的问题：

1. 按原公式重算TP4，已计逻辑工作中HBM需求时间是否仍高于Tensor Core需求时间？这是公式内部的条件判断。
2. 这些资源需求是否足以解释真实整模型decode延迟和主要算子？这是模型覆盖及运行时验证。
3. 同一形状下，实际改变HBM频率或核心频率，延迟对哪一项更敏感？只有生效的干预才能回答；核心频率不只影响Tensor Core。

原公式的BS1、OSL32、Engram查表表驻留主机、FP8索引与NVFP4主KV条件保持不变，仅把TP/EP从8改为4。理论TC/SFU/HBM效率假设70%，CC为80%，参考频率1965 MHz；这些不是实测利用率或校准后的设备速率。公式缺失或不完整覆盖top-k、mHC、norm、Engram执行/主机路径、通信、启动及中间流量，不作为完整延迟预测。

## 运行时与测量形状

vLLM `0.29.1rc1.dev573+ge97573215`，Python前端；DeepSeek-V4.1-Flash，TP4/EP4，Mega后端`FLASHMLA_MEGA_ATTN_DSV41`，NVFP4主KV、FP8索引，128-token block；`FULL_AND_PIECEWISE`，max-num-seqs=256，graph capture上限64，chunk-prefill=16384，模型上下文上限262144。版本、后端与布局由启动日志核对；使用任务自有入口采集数值，不修改安装源码或模型计算。

| 测量组 | Prompt token N | 实际目标batch | 正式输出OSL | 用途 |
|---|---|---|---|---|
| 原理论对齐 | 8192/65536/131072 | 1 | 32 | 对照BS1需求模型 |
| Batch扩展 | 8192 | 16/48/64 | 128 | 检查实际decode batch扩展，不冒充原BS1公式 |
| 频率敏感性 | 65536；8192 | 分别1；64 | 256 | 同形状、同输入/输出预算的干预对照 |
| 独立profiler | 65536；8192 | 分别1；64 | 64 | 每卡采集8个纯decode步骤，不进入正常计时均值 |

使用固定随机种子20261021生成相同token ID输入，属于合成形状实验，不是AgentX或真实任务新增重复。每个测点先清空缓存并验证success=true，再准备N−128个历史token，随后追加128个token正式输出；准备阶段输出1只用于建历史，正式输出不是1。每请求核对prompt、输出及恰好N−128个缓存token。

共3次全新服务，服务间不复用KV；每服务内测点也显式重置并使用独立salt。理论/扩展测点随机排序；频率条件按固定随机顺序执行，每个形状另补一个reference对照。编译缓存可以复用，不等于复用KV。

另补一套全新服务，做3个服务内block、共18个固定生成token的频率对照。原因是正常生成即使temperature=0、输入及OSL相同，实测输出序列仍有差异；仅按输出长度对齐不能排除后续上下文内容变化。诊断组以allowed_token_ids=[123]限制每个输出位置，仍执行256个完整生成步骤，并检查每个请求的实际token序列恰好一致。它排除输出序列不同这一变量，不保证所有浮点中间结果或专家路由位级相同；属于受限输出的合成控制，不代表自然文本生成或质量评估。

该补充组只有1次独立服务、3个服务内block，不能算成额外3次独立服务；与前述3次正常生成服务分开统计，不合并为更大独立样本量。

## 频率干预必须核验实际值

对本实验四卡尝试reference核心1965 MHz、低核心1200 MHz，以及HBM由3996降至3402 MHz。每个测点采集100 ms设备频率/功耗趋势，并按纯decode时间窗口检查实际频率。reference实际核心约1852 MHz，应使用实测值，而不是把请求的1965 MHz当作已生效值。

本机即时HBM锁频接口返回“不支持，请用deferred”，即使命令退出码为0，实测HBM仍3996 MHz。该组标为“HBM请求未生效”，只保留为无效干预/同频对照，不能用其延迟变化估算HBM弹性。deferred模式提示需要后续GPU初始化，驱动重载可保证生效；本轮没有重载共享服务器驱动或重置其他GPU。

只有低核心条件实际降至1200 MHz。它同时影响Tensor/CUDA核心、mHC、控制和通信kernel等路径，不能称为独立减少Tensor Core算力。结束或异常退出恢复本实验设备的核心默认设置；不改变其他GPU的时钟。

## 计时与profile口径

正常性能主指标为后端iteration elapsed_ms；筛选context request/token均为0、generation request/token均等于目标batch的非dummy步骤。它包含该迭代执行及宿主侧关键路径，不是纯GPU CUDA-event时间；不能直接拿客户端流式TPOT或理论已计GPU子项替代。保留所有匹配步骤及去掉最初两步的辅助统计。

每次服务先独立汇总步骤均值，再对3个服务求均值和运行间Min–Max；不把数百步骤伪装成数百次独立实验。频率对比以同一服务、同一形状的两个reference均值作base。3次重复仅描述本次范围，不给置信区间或显著性断言。

Profiler独立运行，四个rank分别分析。一个execute annotation会在多个CUDA stream上重复出现，先按External id去重得到8个步骤；不把56/72条GPU范围误算为56/72步。kernel按名字保守分类为专家GEMM、其他GEMM、融合attention、索引扫描/选择、mHC、通信等，未能确定的保留其他项；不把kernel名字当作内存带宽计数器。

Kernel duration之和允许跨stream重叠，只用于“累计kernel时长”构成，不叫整步墙钟占比；设备活动时间使用kernel/memcpy/memset区间并集。通信kernel的时间可能含等待，不能直接归因于链路带宽不足；无设备活动的间隙也不能未经宿主跟踪就归因于CPU瓶颈。

## 发布与后续

公开脱敏测点、逐rank汇总、理论子项、分析脚本及原始材料指纹；原始服务日志和trace由实验所有者保留，不公开部署路径、设备UUID、请求ID或token ID。运行中检查同卡其他进程；共享CPU、主存等背景干扰仍是局限。

本轮能检验理论覆盖、核心频率敏感性与算子结构，但无有效HBM调频时，不能确定“增加带宽比增加计算更划算”。完整HBM/主机链路计数器、状态容量/恢复E2、等预算PD路由E3与ASIC约束下设计空间E4仍需另行验证。
