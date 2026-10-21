# 环境、版本与来源边界

## AgentX 回放

B200×4、TP4、正常前缀缓存；最终测点 C16/32/48/56/64，`max-num-seqs=256`。所有点独立启动新服务，首次未命中计入，无请求预热。固定 256 个会话、前 8 轮；模型的有效请求数及上下文限制见[数据集表](../tables/dataset.md)。

Qwen 的已记录启动参数包括 BF16、上下文 262144、`max-num-batched-tokens=16384`、`gpu-memory-utilization=0.9`、chunked prefill、prefix caching。不能把该组参数套用到另外两个模型。原始模型权重和运行目录不公开。

跨模型 tokenizer/template、有效样本、运行栈存在差异。V4 与 V4.1 的共同 trace 清单指纹相同，输入 token 总量略有不同；Qwen 的过滤后清单另有指纹。每个点的请求指标及清单 SHA256 保存在 [agentx.json](../data/agentx.json)。

## 正式真实任务：实际安装版本

| 模型 | vLLM 安装版本 | OCI 记录的源码 revision | Claude Code |
|---|---|---|---|
| DeepSeek-V4.1-Flash | 0.29.1rc1.dev573+ge97573215 | e9757321527ca1ecd514c07c1418dd2c53da3d19 | 2.1.241 |
| DeepSeek-V4-Flash-0731 | 0.29.1rc1.dev573+ge97573215 | e9757321527ca1ecd514c07c1418dd2c53da3d19 | 2.1.241 |
| Qwen3.8-27B | 0.26.0 | ffd46bfab2128bb84146050e98b51a617c6575ab | 2.1.241 |

| 核验对象 | DeepSeek 两个服务 | Qwen 派生镜像 |
|---|---|---|
| 镜像 SHA256 | f75fec992c293dd41042745f8aa71bfdf12b188b6a01e0baf075e4bca5d297e5 | 770fe65b2c73ee74a5c42165cf3433de4048cc2cd9c57a937ca4e35aba5aa87b |
| 已安装 Anthropic serving.py SHA256 | 97817be55f9fc8a2a9551994d2675e10e9f2fa253c1c48d2083d1c111faa638d | 9b3386c394d500820f5777a1c000f87e2c3672734da00a5d62e599fd7ae08e59 |

以上信息来自原正式服务容器的只读核验，不是按镜像标签猜版本。Qwen 是派生镜像，OCI revision 不能证明每个文件等同于官方对应提交；本报告不泛称所有官方 vLLM 0.26.0 都具有同样行为。原始配置和源码由所有者受控保存，仓库公开身份指纹及分析结果，不公开部署拓扑。

## 消融

V4.1 的 C4 Graph 消融保持模型、镜像、TP4/EP4、Mega Attention、`nvfp4_ds_mla` KV 格式等参数不变，改变 `FULL_AND_PIECEWISE`/`FULL_DECODE_ONLY`。使用 vLLM `0.29.1rc1.dev573+ge97573215`。服务端首次调度到首 token 的统计边界见正文第3节，不是纯 kernel 时间。

## 新增TP4资源配比验证

沿用DeepSeek镜像`f75fec992c293dd41042745f8aa71bfdf12b188b6a01e0baf075e4bca5d297e5`，vLLM `0.29.1rc1.dev573+ge97573215`。B200×4、TP4/EP4，Python前端、Mega、NVFP4主KV/FP8索引、128-token block、`FULL_AND_PIECEWISE`，max-num-seqs=256，capture上限64，上下文上限262144。它与早期AgentX上下文上限不同，不混作原回放新增重复。

理论固定到model_analysis main的`f2b5d167a75c2bb6a4b9385d2fd2de30f6b5278a`，不使用服务器上未提交的BS8公式。资源速率假设不是校准后的设备性能。核心请求1965 MHz、实测约1852 MHz；低核心实际1200 MHz；即时HBM锁频不支持，实测一直3996 MHz。计时、独立profile与固定输出控制的不同重复单位见[协议](decode-theory-tp4-protocol.md)。

## 新版客户端的独立行为验证

| 项目 | 核验值 |
|---|---|
| 实测客户端 | Claude Code 2.1.280 |
| 客户端二进制 SHA256 | 1e08503dbdf3c2cb0d706d32f3408277388d1c76ef108673e8fe42c1b322925b |
| 网关 | LiteLLM 1.98.0 |
| 后端版本接口 | vLLM 0.28.1rc1.dev688+g5c642796e |
| 后端接口报告的镜像 SHA256 | 6ffc2b4c8b6b3f3962ca4602f02a5b49593432a6a407a361639b587c457f3ff0 |

这是与 B200 正式任务分开的链路。上述后端身份来自只读接口，不是容器安装包审计；“A800”仅为已有用户配置的入口别名，物理 GPU 和具体 worker/pod 未独立核验。同一个网关 deployment id 也不证明内部始终同一个 worker。

该验证统一最多输出16 token，仅确认通知相关缓存行为，不参与性能排名。关闭逐轮 token 提醒仅发生于临时客户端配置，不修改正式实验结果。没有根据“最新版客户端”推断“最新版服务端已经或尚未修复”。
