#!/usr/bin/env python3
"""Generate the numeric TP4 tables from allowlisted evidence."""
import json
from pathlib import Path
import statistics
from plot_decode_theory import paired

ROOT=Path(__file__).resolve().parents[1]


def fmt(values,digits=3):
    return '/'.join(f'{v:.{digits}f}' for v in values)


def main():
    data=json.loads((ROOT/'data/decode_theory_tp4.json').read_text())
    lines=['# V4.1-Flash：B200四卡资源配比假说验证','',
           '合成形状实验，TP4/EP4，3次全新服务；不属于AgentX或真实任务新增重复。',
           '正常计时和独立profiler分开；主指标为实际匹配batch的后端decode迭代耗时，不是纯GPU CUDA-event时间。','',
           '## 1. BS1、OSL32：部分理论与实测','',
           '| Prompt token | 已计TC(ms) | 已计HBM(ms) | 已计GPU合成(ms) | 后端步骤(ms) | 3次步骤均值(ms) |',
           '|---|---:|---:|---:|---:|---|']
    for n in (8192,65536,131072):
        theory=[r for r in data['theory'] if r['inputTokens']==n]
        summary=next(r for r in data['summaries'] if r['label']=='theory_bs1' and r['inputTokens']==n)
        total=lambda k:sum(r[k] or 0 for r in theory)
        lines.append(f"| {n} | {total('TC_ms'):.6f} | {total('HBM_ms'):.6f} | {total('block_known_ms'):.6f} | {summary['stepMs']['mean']:.3f} | {fmt(summary['runStepMeansMs'])} |")
    lines+=['','TC、HBM为原公式和效率假设下已计需求；合成逐模块max(TC,HBM)+CC+SFU，缺失不补零。',
            '该模型明确不完整，不能把实测减去理论的差额直接命名为“通信耗时”或“遗漏占比”，也不能称为完整预测误差。','',
            '## 2. 有效核心频率干预','',
            '| Batch/Prompt | reference步骤(ms) | 低核心步骤(ms) | 平均增加(%) | 3次配对增加(%) | 配对增加范围(%) |',
            '|---|---:|---:|---:|---|---|']
    for n,b in ((65536,1),(8192,64)):
        rows=paired(data,n,b)
        changes=[r['changePercent'] for r in rows]
        lines.append(f"| {b}/{n} | {statistics.fmean(r['reference'] for r in rows):.3f} | {statistics.fmean(r['lower'] for r in rows):.3f} | {statistics.fmean(changes):.2f} | {fmt(changes,2)} | {min(changes):.2f}–{max(changes):.2f} |")
    lines+=['','OSL256。每次服务reference是同形状两个正常核心对照的均值；范围是3次服务的配对变化范围，不是置信区间。',
            '正常生成的输入及输出长度相同，但生成token序列不同，不能当作逐token内容完全一致的控制。下表另以固定输出token补充核验。',
            '实际核心约1852→1200 MHz（下降35.21%），HBM保持3996 MHz。核心频率影响的不只有Tensor Core，不能据此认定Tensor Core算力是唯一瓶颈。',
            '请求HBM=3402 MHz的6个测点均未生效，实际仍3996 MHz；不计算HBM带宽弹性，不与有效核心干预比较资源投入收益。','',
            '### 固定生成token的补充对照','',
            '| Batch/Prompt | reference步骤(ms) | 低核心步骤(ms) | 平均增加(%) | 3个block增加(%) |',
            '|---|---:|---:|---:|---|']
    for n,b in ((65536,1),(8192,64)):
        rows=paired(data,n,b,fixed=True)
        changes=[r['changePercent'] for r in rows]
        lines.append(f"| {b}/{n} | {statistics.fmean(r['reference'] for r in rows):.3f} | {statistics.fmean(r['lower'] for r in rows):.3f} | {statistics.fmean(changes):.2f} | {fmt(changes,2)} |")
    lines+=['','一个额外全新服务内3个block，各shape包含两个reference及一个低核心对照；共18个测点，OSL256。百分比先按block配对再平均，不由跨block均值直接相除。',
            'allowed_token_ids=[123]，逐请求核验输入和完整输出序列。此组不是3次独立服务，不与自然生成混合统计，也不代表真实文本质量。','',
            '## 3. 主实验：未开启profiler的计时','',
            '| 测量组 | Prompt | Batch | OSL | 3次步骤均值(ms) | 均值(ms) | 运行间范围(ms) |',
            '|---|---:|---:|---:|---|---:|---|']
    labels={'theory_bs1':'理论对齐','batch_extension':'Batch扩展','sensitivity_reference':'频率对照',
            'sensitivity_reference_post':'频率后置对照','sensitivity_core1200':'低核心1200 MHz',
            'sensitivity_memory3402':'HBM请求未生效：同频对照'}
    for row in data['summaries']:
        st=row['stepMs']
        lines.append(f"| {labels[row['label']]} | {row['inputTokens']} | {row['batch']} | {row['outputTokens']} | {fmt(row['runStepMeansMs'])} | {st['mean']:.3f} | {st['min']:.3f}–{st['max']:.3f} |")
    lines+=['','不同OSL/形状不能混合求均值。只保留context=0且实际generation batch/token与目标相等的非dummy步骤，逐测点样本量及频率分布见公开JSON。','',
            '## 4. 独立profile：累计kernel时间，不是墙钟占比','',
            '| Kernel分类 | BS1/65536(ms/step) | 累计kernel占比(%) | BS64/8192(ms/step) | 累计kernel占比(%) |',
            '|---|---:|---:|---:|---:|']
    categories=[('moe_expert_gemm','专家GEMM'),('other_gemm','其他GEMM'),('mhc','mHC'),
                ('attention_fused','融合attention'),('index_scan_selection','索引扫描/选择'),
                ('collectives','通信kernel'),('norm_position_quant','norm/位置/量化'),
                ('engram_lookup','Engram lookup'),('moe_route_combine','MoE路由/合并'),('other','其他')]
    for key,label in categories:
        cells=[]
        for b in (1,64):
            rows=[r for r in data['profiles'] if r['rank']==0 and r['batch']==b]
            values=[r['kernelGroupsMs'].get(key,0)/r['decodeSteps'] for r in rows]
            value=statistics.fmean(values)
            total=statistics.fmean(r['kernelDurationSumMs']/r['decodeSteps'] for r in rows)
            cells += [f'{value:.3f}({min(values):.3f}–{max(values):.3f})',f'{100*value/total:.2f}']
        lines.append('| '+' | '.join([label,*cells])+' |')
    lines+=['','此表只展示rank0的3次profile均值(范围)，每次8个步骤；分类互斥，但跨stream的kernel时间允许重叠，不能解释为完整步骤的互斥耗时分解。',
            'mHC包含hc_prenorm GEMM；其他GEMM未全部映射为主投影，MoE专家GEMM也不是整个MoE模块。两个profile的上下文不同，不属于只改变batch的消融。','',
            '| Profile形状 | Rank | 设备活动并集/span(%) | 通信kernel(ms/step) |',
            '|---|---:|---:|---:|']
    for b in (1,64):
        for rank in range(4):
            rows=[r for r in data['profiles'] if r['rank']==rank and r['batch']==b]
            busy=statistics.fmean(100*r['activityUnionMs']/r['spanMs'] for r in rows)
            values=[r['kernelGroupsMs'].get('collectives',0)/r['decodeSteps'] for r in rows]
            comm=statistics.fmean(values)
            lines.append(f'| BS{b} | {rank} | {busy:.2f} | {comm:.3f}({min(values):.3f}–{max(values):.3f}) |')
    lines+=['','span为首个至末个pure-decode GPU execute范围包络；活动并集包含kernel、memcpy、memset。',
            '通信kernel可能含等待；rank差异不等于已证明网络带宽受限。无活动间隙也不是直接证明CPU瓶颈。',
            '', '[验证协议](../docs/decode-theory-tp4-protocol.md) · [数值与指纹](../data/decode_theory_tp4.json)','']
    (ROOT/'tables/decode_theory_tp4.md').write_text('\n'.join(lines))


if __name__=='__main__':
    main()
