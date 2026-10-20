#!/usr/bin/env python3
"""Render the standalone measured FlashMLA API and counter evidence."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({'svg.fonttype':'none','svg.hashsalt':'agent-workload-serving-study',
                     'font.family':'DejaVu Sans','savefig.facecolor':'white'})


def main():
    data = json.loads((ROOT/'data/decode_attention_microbench.json').read_text())
    batches = (1,16,48,64)
    fmts = (('v41_fp8','FP8, 528 B/record','#E97F68'),
            ('v41_nvfp4','NVFP4, 288 B/record','#AF2E3D'))
    summaries = {(s['format'],s['cacheMode'],s['historyLength'],s['batch']):s['latencyMs']
                 for s in data['summaries']}
    counters = {(s['format'],s['batch']):s['metrics'] for s in data['counterSummaries']}
    fig, axes = plt.subplots(1,2,figsize=(12,4.9))
    for fmt, label, color in fmts:
        values = [summaries[(fmt,'cold',65536,b)] for b in batches]
        mean = [v['mean']*1000 for v in values]
        errors = [[(v['mean']-v['min'])*1000 for v in values],
                  [(v['max']-v['mean'])*1000 for v in values]]
        axes[0].errorbar(batches,mean,yerr=errors,fmt='o-',ms=3,capsize=3,color=color,label=label)
        values = [counters[(fmt,b)]['dram__bytes_read.sum'] for b in batches]
        mean = [v['mean']/1e6 for v in values]
        errors = [[(v['mean']-v['min'])/1e6 for v in values],
                  [(v['max']-v['mean'])/1e6 for v in values]]
        axes[1].errorbar(batches,mean,yerr=errors,fmt='o-',ms=3,capsize=3,color=color,label=label)
    axes[0].set(ylabel='FlashMLA API latency (us)',title='Unprofiled CUDA-graph latency',ylim=(27,35))
    axes[1].set(ylabel='Measured DRAM read (MB, decimal)',title='Separate main-kernel hardware counters',ylim=(0,34))
    for ax in axes:
        ax.set_xlabel('Batch (sequences)')
        ax.set_xticks(batches)
        ax.grid(axis='y',alpha=.22)
        ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('B200: fewer KV bytes need not reduce sparse-attention latency',fontsize=14,y=.98)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.91),ncol=2,frameon=False)
    fig.subplots_adjust(top=.72,bottom=.24,wspace=.3)
    fig.text(.07,.045,'L=65,536; selected compressed KV=512 + sliding window=128; 16 live query heads padded to 64.\n'
             'Mean and observed min-max across 3 fresh processes. Cold-cache protocols differ between latency and NCU.\n'
             'Standalone sparse FlashMLA only: not Mega, whole-model decode, task quality or ASIC measurements.',fontsize=8,color='#555555')
    target = ROOT/'figures/12_decode_sparse_microbench'
    fig.savefig(target.with_suffix('.png'),dpi=180,bbox_inches='tight')
    fig.savefig(target.with_suffix('.svg'),bbox_inches='tight',metadata={'Date':'2026-10-20'})
    plt.close(fig)
    svg = target.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    lines = ['# B200稀疏Attention单算子验证','',
             '| Batch | FP8 API(μs) | NVFP4 API(μs) | FP8 DRAM读取(MB) | NVFP4 DRAM读取(MB) | 读取量减少(%) |',
             '|---|---:|---:|---:|---:|---:|']
    for batch in batches:
        f,n = [summaries[(fmt,'cold',65536,batch)] for fmt,_,_ in fmts]
        fb,nb = [counters[(fmt,batch)]['dram__bytes_read.sum']['mean']/1e6 for fmt,_,_ in fmts]
        value = lambda v:f"{v['mean']*1000:.2f}({v['min']*1000:.2f}–{v['max']*1000:.2f})"
        lines.append(f'| {batch} | {value(f)} | {value(n)} | {fb:.2f} | {nb:.2f} | {(1-nb/fb)*100:.2f} |')
    lines.extend(['','API列为三次独立进程均值(运行间Min–Max)，不是置信区间。DRAM是主稀疏kernel实测读取量，MB=10⁶字节；API时间覆盖FlashMLA接口，不与NCU采集时长混用。',
                  '', 'L=65,536，固定选中512个压缩KV及128个滑窗KV，16个有效query head按已安装kernel约束补齐到64。每种格式使用相同随机分布的有限KV，而非同一数值的配对量化；本实验不评估量化质量。FP8/NVFP4记录分别528/288字节。',
                  '', '完整网格包含L=8,192/65,536/131,072、B=1/16/48/64，三种格式/缓存条件各三次进程运行，共108个形状测点；独立NCU采集24个主kernel样本。hot仅包含NVFP4。',
                  '', '结论范围：这批固定top-k、低并行度的独立稀疏attention调用中，减少KV字节没有降低API时间；不能据此认定整模型不受带宽限制，更不能推导ASIC加速比。完整服务E1、E2容量/恢复、E3动态路由和E4建模尚未完成。',''])
    (ROOT/'tables/decode_sparse_microbench.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
