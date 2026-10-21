#!/usr/bin/env python3
"""Allowlisted full-model TP4 measurements versus pinned partial theory.

Read private artifacts; export numbers and fingerprints, never deployment paths,
token IDs, request IDs, messages, GPU UUIDs or raw traces. Profiles are separate
from unprofiled measurements; nested annotations are not summed as components.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


def stats(values):
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError('Missing/nonfinite observations')
    return dict(mean=statistics.fmean(values),min=min(values),max=max(values),count=len(values),
                p50=statistics.median(values))


def union_length(intervals):
    end = None
    result = 0.
    for a,b in sorted(intervals):
        if b<a:
            raise ValueError('Negative interval')
        result += max(0,b-max(a,end if end is not None else a))
        end = max(b,end if end is not None else b)
    return result


def kernel_group(name):
    n=name.lower()
    if any(x in n for x in ('allreduce','all_reduce','nccl','mnnvl')):
        return 'collectives'
    if 'bmm' in n and ('mxe2m1' in n or 'mx_e2m1' in n):
        return 'moe_expert_gemm'
    if any(x in n for x in ('mqa_logits','topk','top_k','sparse_attn_index')):
        return 'index_scan_selection'
    if 'fused_norm_rope_attn_rope_cast' in n or 'sparse_fwd' in n:
        return 'attention_fused'
    if 'mhc' in n or 'hc_prenorm' in n:
        return 'mhc'
    if 'engram' in n or 'lookup' in n:
        return 'engram_lookup'
    if any(x in n for x in ('moe','router','finalize')):
        return 'moe_route_combine'
    if any(x in n for x in ('gemm','gemv','nvjet')):
        return 'other_gemm'
    if any(x in n for x in ('norm','rope','quant','cast')):
        return 'norm_position_quant'
    return 'other'


def profile_summary(source):
    events=json.load(gzip.open(source,'rt'))['traceEvents']
    ranges=[e for e in events if e.get('cat')=='gpu_user_annotation' and
            re.match(r'execute_\d+_context_0\(',e.get('name','')) and '_generation_' in e['name']]
    if not ranges:
        raise ValueError('No confirmed pure-decode GPU ranges')
    shapes={tuple(map(int,re.match(r'execute_(\d+)_context_0.*_generation_(\d+)',e['name']).groups())) for e in ranges}
    if len(shapes)!=1 or next(iter(shapes))[0]!=next(iter(shapes))[1]:
        raise ValueError('Profile mixes decode batch shapes')
    batch=next(iter(shapes))[0]
    # CUPTI emits the same external annotation once per CUDA stream. Eight
    # execute calls can therefore appear as 56/72 ranges, not 56/72 steps.
    grouped=defaultdict(list)
    for event in ranges:
        grouped[event.get('args',{}).get('External id',event['name'])].append(event)
    ranges=[dict(ts=min(e['ts'] for e in group),
                 dur=max(e['ts']+e['dur'] for e in group)-min(e['ts'] for e in group))
            for group in grouped.values()]
    start=min(e['ts'] for e in ranges)
    stop=max(e['ts']+e['dur'] for e in ranges)
    kernels=[e for e in events if e.get('cat')=='kernel' and e.get('dur',0)>0 and
             e['ts']<stop and e['ts']+e['dur']>start]
    groups=defaultdict(float)
    names=defaultdict(float)
    intervals=[]
    for event in kernels:
        a,b=max(start,event['ts']),min(stop,event['ts']+event['dur'])
        groups[kernel_group(event['name'])]+=(b-a)/1000
        names[event['name']]+=(b-a)/1000
        intervals.append((a,b))
    activity=[(max(start,e['ts']),min(stop,e['ts']+e['dur'])) for e in events
              if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset') and e.get('dur',0)>0 and
              e['ts']<stop and e['ts']+e['dur']>start]
    span=(stop-start)/1000
    busy=union_length(activity)/1000
    if busy>span+1e-6:
        raise ValueError('Invalid activity union')
    match=re.search(r'_rank(\d+)\.',source.name)
    return dict(rank=int(match.group(1)),batch=batch,decodeSteps=len(ranges),
                executeRangeMs=stats([e['dur']/1000 for e in ranges]),spanMs=span,
                kernelUnionMs=union_length(intervals)/1000,activityUnionMs=busy,
                noRecordedDeviceActivityMs=span-busy,kernelDurationSumMs=sum(groups.values()),
                kernelCount=len(kernels),kernelGroupsMs=dict(groups),
                largestKernels=[dict(name=n,sumMs=t) for n,t in sorted(names.items(),key=lambda x:-x[1])[:12]])


def read_iterations(source):
    rows=[]
    for line in source.open():
        if 'DECODE_ITERATION_METRICS ' not in line:
            continue
        row=json.loads(line.split('DECODE_ITERATION_METRICS ',1)[1])
        details=row.get('iteration_details')
        if details and not details.get('is_dummy'):
            rows.append(dict(details,logged_unix=row['logged_unix']))
    return rows


def measured_case(directory, iterations, telemetry):
    config=json.loads((directory/'case.json').read_text())
    data=json.loads((directory/'client.json').read_text())
    audit=json.loads((directory/'audit.json').read_text())
    rows=[r for r in data['rows'] if r['stage']=='timed']
    n=config['history_length']+config['append_length']
    b,o=config['batch'],config['output_tokens']
    if data['status']!='complete' or not audit['decode_isolation_valid'] or len(rows)!=b:
        raise ValueError('Incomplete or cache-invalid case')
    if any(r['error'] or r['output_tokens']!=o or r['prompt_tokens']!=n or
           r['cached_tokens']!=config['history_length'] for r in rows):
        raise ValueError('Incorrect token/cache shape')
    if config.get('fixed_output_token_id') is not None and any(
            r['output_token_ids'] != [config['fixed_output_token_id']]*o for r in rows):
        raise ValueError('Fixed-output tokens differ from the protocol')
    begin=min(r['started_unix'] for r in rows)
    end=max(r['finished_unix'] for r in rows)
    pure=[r for r in iterations if begin<=r['logged_unix']<=end+.1 and
          r['num_ctx_requests']==0 and r['num_ctx_tokens']==0 and
          r['num_generation_requests']==b and r['num_generation_tokens']==b]
    if len(pure)<16:
        raise ValueError(f'Insufficient matched actual-batch decode steps in {directory.name}')
    if any(r['elapsed_ms']<=0 for r in pure):
        raise ValueError('Nonpositive backend iteration time')
    clocks=defaultdict(lambda:defaultdict(list))
    active_start=min(r['logged_unix']-r['elapsed_ms']/1000 for r in pure)
    active_end=max(r['logged_unix'] for r in pure)
    for record in csv.reader(telemetry.open()):
        if len(record)!=8:
            continue
        stamp,index,sm,mem,util,memutil,power,temp=map(str.strip,record)
        ts=datetime.strptime(stamp,'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc).timestamp()
        if active_start-.05<=ts<=active_end+.05:
            for key,val in (('smMHz',sm),('memoryMHz',mem),('gpuUtilPercent',util),
                            ('memoryUtilPercent',memutil),('powerW',power),('temperatureC',temp)):
                clocks[int(index)][key].append(float(val))
    if len(clocks)!=4:
        raise ValueError('Missing four-device in-decode telemetry')
    requested_mem=3402 if config['clock_condition']=='memory3402' else 3996
    memory_applied=all(all(x==requested_mem for x in d['memoryMHz']) for d in clocks.values())
    return dict(id=directory.name,repeat=config['repeat'],inputTokens=n,batch=b,outputTokens=o,
                cachedTokensPerRequest=config['history_length'],profile=bool(config.get('profile')),
                condition=config['clock_condition'],pureDecodeSteps=len(pure),
                inputSequenceDigest=hashlib.sha256(json.dumps([r['input_sha256'] for r in rows]).encode()).hexdigest(),
                outputSequenceDigest=hashlib.sha256(json.dumps([r['output_token_ids'] for r in rows]).encode()).hexdigest(),
                requestedMemoryMHz=requested_mem,memoryClockRequestApplied=memory_applied,
                memoryInterventionValid=(config['clock_condition']=='memory3402' and memory_applied),
                stepMs=stats([r['elapsed_ms'] for r in pure]),
                steadyAfterTwoStepsMs=stats([r['elapsed_ms'] for r in pure[2:]]),
                clocks=[dict(rank=i,**{k:stats(v) for k,v in values.items()})
                        for i,(_,values) in enumerate(sorted(clocks.items()))])


def frequency_comparisons(cases):
    groups=defaultdict(list)
    for row in cases:
        if row['outputTokens']==256 and not row['profile']:
            groups[(row['repeat'],row['inputTokens'],row['batch'])].append(row)
    output=[]
    for (repeat,n,b),rows in sorted(groups.items()):
        refs=[r['stepMs']['mean'] for r in rows if r['condition']=='reference']
        core=[r['stepMs']['mean'] for r in rows if r['condition']=='core1200']
        if len(refs)!=2 or len(core)!=1:
            continue
        reference=statistics.fmean(refs)
        output.append(dict(repeatOrBlock=repeat,inputTokens=n,batch=b,referenceStepMs=reference,
             lowerCoreStepMs=core[0],increasePercent=100*(core[0]/reference-1),
             identicalInputs=len({r['inputSequenceDigest'] for r in rows})==1,
             identicalGeneratedOutputs=len({r['outputSequenceDigest'] for r in rows})==1))
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args()
    root=args.root
    experiment=root/'theory_validation_20261021'
    status=json.loads((experiment/'status.json').read_text())
    if status['state']!='complete' and not args.allow_partial:
        raise ValueError('Experiment not complete')
    evidence={}
    def fingerprint(path):
        evidence[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    cases=[]
    profiles=[]
    versions=set()
    images=set()
    for repeat in range(1,4):
        service=root/f'services/v41_theory_r{repeat}'
        log=service/'server.log'
        if not log.exists():
            continue
        text=log.read_text()
        ready=service/'ready.json'
        if ready.exists():
            images.add(json.loads(ready.read_text())['image'])
            fingerprint(ready)
            fingerprint(service/'launch.json')
        versions.update(re.findall(r'Initializing a V1 LLM engine \(v([^)]*)\)',text))
        iterations=read_iterations(log)
        for directory in sorted((experiment/'cases').glob(f'*_r{repeat}')):
            if not (directory/'clock.json').exists():
                continue
            telemetry=experiment/'telemetry'/(directory.name+'.csv')
            row=measured_case(directory,iterations,telemetry)
            cases.append(row)
            for p in (directory/'case.json',directory/'client.json',directory/'audit.json',directory/'clock.json',telemetry):
                fingerprint(p)
        fingerprint(log)
        for source in sorted((root/'profiles'/f'v41_theory_r{repeat}').glob('*.gz')):
            item=profile_summary(source)
            item['repeat']=repeat
            profiles.append(item)
            fingerprint(source)
    theory=[]
    source=experiment/'model_tp4/blocks.csv'
    for row in csv.DictReader(source.open()):
        if row['phase']!='decode' or row['model']!='DeepSeek-V4.1-Flash' or int(row['input_tokens']) not in (8192,65536,131072):
            continue
        theory.append(dict(inputTokens=int(row['input_tokens']),block=row['block'],
                      **{k:float(row[k]) if row[k] else None for k in ('TC_ms','CC_ms','SFU_ms','HBM_ms','block_known_ms')}))
    fingerprint(source)
    for source in (experiment/'profile.json',experiment/'profile-device-probe.json',
                   experiment/'source/data/module-skills/modelmeter-analysis-v2/scripts/estimate_deepseek_comparison.py',
                   root/'scripts/theory_driver.py',root/'scripts/service_control.py',root/'scripts/decode_instrumented_entry.py',
                   root/'scripts/decode_probe_client.py',experiment/'clock_restore.json'):
        if source.exists():
            fingerprint(source)
    groups=defaultdict(list)
    for case in cases:
        if not case['profile']:
            key=(case['id'].split('_n',1)[0],case['inputTokens'],case['batch'],case['outputTokens'])
            groups[key].append(case)
    summaries=[dict(label=k[0],inputTokens=k[1],batch=k[2],outputTokens=k[3],
                    runStepMeansMs=[r['stepMs']['mean'] for r in sorted(v,key=lambda x:x['repeat'])],
                    stepMs=stats([r['stepMs']['mean'] for r in v])) for k,v in sorted(groups.items())]
    if not args.allow_partial and (len(cases)!=48 or len(profiles)!=24 or any(s['stepMs']['count']!=3 for s in summaries)):
        raise ValueError('Expected 48 cases, 24 rank profiles and 3 independent service runs')
    if len(versions)!=1 or len(images)!=1:
        raise ValueError('Unpinned/mixed runtime versions or images')
    fixed=root/'fixed_token_validation_20261021'
    fixed_cases=[]
    fixed_status={'state':'not_started'}
    if (fixed/'status.json').exists():
        fixed_status=json.loads((fixed/'status.json').read_text())
        if fixed_status['state']!='complete' and not args.allow_partial:
            raise ValueError('Fixed-token follow-up not complete')
        log=root/'services/v41_fixed_followup/server.log'
        if log.exists():
            iterations=read_iterations(log)
            for directory in sorted((fixed/'cases').glob('*')):
                if not (directory/'clock.json').exists():
                    continue
                telemetry=fixed/'telemetry'/(directory.name+'.csv')
                row=measured_case(directory,iterations,telemetry)
                row['repetitionUnit']='within-service block; one additional fresh service total'
                fixed_cases.append(row)
                for path in (directory/'case.json',directory/'client.json',directory/'audit.json',telemetry):
                    fingerprint(path)
            fingerprint(log)
            fingerprint(root/'scripts/fixed_token_driver.py')
            if (fixed/'clock_restore.json').exists():
                fingerprint(fixed/'clock_restore.json')
    if not args.allow_partial and len(fixed_cases)!=18:
        raise ValueError('Expected 18 fixed-token follow-up cases, three blocks in ONE additional service')
    result=dict(schemaVersion=1,platform='NVIDIA B200',gpus=4,tp=4,ep=4,
                model='DeepSeek-V4.1-Flash',vllmVersion=next(iter(versions)),imageSha256=next(iter(images)).removeprefix('sha256:'),
                theoryRepository='https://github.com/Romaosir/model_analysis',theoryCommit=status['theory_commit'],
                theoryEfficiencies=dict(TC=.7,HBM=.7,CC=.8,SFU=.7),
                theoryStatus='Partial and uncalibrated; counted resource-demand times, not full latency or measured utilization',
                theoryOmissions=['mHC/norm arithmetic','index selection','Engram host path and execution',
                                 'communication and launch','incomplete activation/intermediate traffic'],
                scope='Synthetic matched-shape TP4 service probes; not AgentX, PD or ASIC performance',
                latencyDefinition='Backend scheduler iteration elapsed_ms, matched actual generation batch; includes execution/host critical path, not pure GPU CUDA-event time',
                profileDefinition='Separate torch profiles, eight pure-decode steps per rank; kernel sum can overlap. Activity uses interval union, not nested annotation sums.',
                clockDefinition='Core-frequency intervention plus an unsuccessful HBM clock request; actual in-decode telemetry retained. Core clocks affect more than Tensor Cores.',
                status=status['state'],fixedTokenStatus=fixed_status['state'],cases=cases,summaries=summaries,profiles=profiles,theory=theory,
                frequencyComparisons=frequency_comparisons(cases),
                fixedTokenCases=fixed_cases,fixedTokenComparisons=frequency_comparisons(fixed_cases),
                fixedTokenProtocol=dict(independentServices=1,withinServiceBlocks=3,outputToken=123,
                    boundary='Diagnostic constrained-token workload, not natural generation or three independent services'),
                evidenceSha256=evidence)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(cases=len(cases),profiles=len(profiles),fixedTokenCases=len(fixed_cases),status=result['status'],summaries=len(summaries))))


if __name__=='__main__':
    main()
