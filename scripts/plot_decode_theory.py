#!/usr/bin/env python3
"""TP4 partial-demand theory, measured core sensitivity and separate profiles."""
from collections import defaultdict
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'svg.fonttype':'none','svg.hashsalt':'agent-workload-serving-study',
                    'font.family':'DejaVu Sans','savefig.facecolor':'white'})


def paired(data,n,b,fixed=False):
    output=[]
    for repeat in (1,2,3):
        rs=[r for r in data['fixedTokenCases' if fixed else 'cases'] if r['repeat']==repeat and r['inputTokens']==n and r['batch']==b and
            r['outputTokens']==256 and not r['profile']]
        reference=[r['stepMs']['mean'] for r in rs if r['condition']=='reference']
        lower=[r['stepMs']['mean'] for r in rs if r['condition']=='core1200']
        if reference and lower:
            output.append(dict(repeat=repeat,reference=statistics.fmean(reference),lower=lower[0],
                               changePercent=100*(lower[0]/statistics.fmean(reference)-1)))
    return output


def main():
    data=json.loads((ROOT/'data/decode_theory_tp4.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(16.5,5.7))
    ns=(8192,65536,131072)
    rows=[next(s for s in data['summaries'] if s['label']=='theory_bs1' and s['inputTokens']==n) for n in ns]
    vals=[r['stepMs'] for r in rows]
    means=[r['mean'] for r in vals]
    err=[[r['mean']-r['min'] for r in vals],[r['max']-r['mean'] for r in vals]]
    theory=[sum(r['block_known_ms'] or 0 for r in data['theory'] if r['inputTokens']==n) for n in ns]
    measured=axes[0].errorbar(range(3),means,yerr=err,fmt='o-',ms=3,capsize=3,color='#AF2E3D',label='Measured backend step')
    theoretical,=axes[0].plot(range(3),theory,'s--',ms=3,color='#BDA9C4',label='Partial counted GPU model')
    axes[0].set(xticks=range(3),xticklabels=['8,192','65,536','131,072'],xlabel='Prompt tokens (BS1, OSL32)',
                ylabel='Time per decode step (ms)',title='Partial theory is not full latency',ylim=(0,max(means)*1.22))
    axes[0].legend(handles=[measured,theoretical],loc='center right',fontsize=8,frameon=False)
    for i,m in enumerate(means):
        axes[0].annotate(f'{m:.2f}',(i,m),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
    x=np.arange(2)
    comparison=[paired(data,n,b,fixed=True) for n,b in ((65536,1),(8192,64))]
    for shift,key,label,color in ((-.17,'reference','Reference SM (~1852 MHz)','#E97F68'),
                                  (.17,'lower','Lower SM (1200 MHz)','#AF2E3D')):
        values=[[r[key] for r in group] for group in comparison]
        mu=[statistics.fmean(v) for v in values]
        errors=[[m-min(v) for m,v in zip(mu,values)],[max(v)-m for m,v in zip(mu,values)]]
        axes[1].bar(x+shift,mu,width=.32,yerr=errors,capsize=3,color=color,label=label)
    for i,group in enumerate(comparison):
        change=statistics.fmean(r['changePercent'] for r in group)
        top=max(r['lower'] for r in group)
        axes[1].text(i,top+.45,f'+{change:.1f}%',ha='center',fontsize=9)
    axes[1].set(xticks=x,xticklabels=['BS1 / 65,536','BS64 / 8,192'],xlabel='Batch / prompt tokens (OSL256)',
                ylabel='Backend step (ms)',title='Fixed-token core-frequency control',ylim=(0,max(r['lower'] for g in comparison for r in g)*1.24))
    axes[1].legend(loc='upper left',fontsize=8,frameon=False)
    groups=[('other_gemm','Other GEMMs','#E97F68'),('moe_expert_gemm','MoE expert GEMMs','#CB5979'),
            ('attention_fused','Fused attention','#AF2E3D'),('mhc','mHC','#BDA9C4'),
            ('collectives','Collectives','#707985'),('remainder','Remaining kernels','#EEDCE1')]
    bottom=np.zeros(2)
    for group,label,color in groups:
        values=[]
        for batch in (1,64):
            profiles=[p for p in data['profiles'] if p['rank']==0 and p['batch']==batch]
            selected={g for g,_,_ in groups if g!='remainder'}
            value=statistics.fmean((sum(v for k,v in p['kernelGroupsMs'].items() if k not in selected)
                                     if group=='remainder' else p['kernelGroupsMs'].get(group,0))/p['decodeSteps'] for p in profiles)
            values.append(value)
        axes[2].bar(x,values,bottom=bottom,width=.48,label=label,color=color)
        bottom+=values
    totals=[[p['kernelDurationSumMs']/p['decodeSteps'] for p in data['profiles'] if p['rank']==0 and p['batch']==b] for b in (1,64)]
    axes[2].errorbar(x,bottom,yerr=[[m-min(v) for m,v in zip(bottom,totals)],
                                  [max(v)-m for m,v in zip(bottom,totals)]],fmt='none',capsize=3,color='#707985')
    axes[2].set(xticks=x,xticklabels=['BS1 / 65,536','BS64 / 8,192'],xlabel='Batch / prompt tokens (rank 0)',
                ylabel='Summed kernel duration / step (ms)',title='Separate profiles: not wall-time shares',ylim=(0,max(max(v) for v in totals)*1.18))
    axes[2].legend(loc='upper center',bbox_to_anchor=(.5,-.22),ncol=3,fontsize=7,frameon=False)
    for ax in axes:
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('V4.1-Flash on B200 x4: resource-demand theory versus runtime evidence',fontsize=14,y=.98)
    fig.subplots_adjust(top=.83,bottom=.30,wspace=.33)
    fig.text(.065,.075,'Left: 3 fresh services. Middle: 3 fixed-token blocks in ONE additional service; not 3 fresh services. '
             'Observed min-max, not confidence intervals.\n'
             'HBM request to 3402 MHz was NOT applied (actual 3996 MHz): no valid HBM-bandwidth intervention. '
             'Lower SM affects Tensor/CUDA cores and other paths.\n'
             'Profiles: 8 decode steps/rank; duration sums can overlap and include profiling overhead. '
             'Synthetic probes only; no PD or ASIC conclusions.',fontsize=8,color='#555555')
    target=ROOT/'figures/13_decode_theory_tp4'
    fig.savefig(target.with_suffix('.png'),dpi=180,bbox_inches='tight')
    fig.savefig(target.with_suffix('.svg'),bbox_inches='tight',metadata={'Date':'2026-10-21'})
    plt.close(fig)
    path=target.with_suffix('.svg')
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


if __name__=='__main__':
    main()
