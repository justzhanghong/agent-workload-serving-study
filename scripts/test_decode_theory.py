#!/usr/bin/env python3
"""CPU-only tests for decode attribution; not independent GPU verification."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from analyze_decode_theory import frequency_comparisons, kernel_group, profile_summary, stats, union_length


class Tests(unittest.TestCase):
    def test_frequency_comparison_and_output_nondeterminism(self):
        rows=[dict(repeat=1,inputTokens=8192,batch=1,outputTokens=256,profile=False,
                   condition=c,stepMs={'mean':v},inputSequenceDigest='same',outputSequenceDigest=str(i))
              for i,(c,v) in enumerate((('reference',10),('reference',14),('core1200',15)))]
        row=frequency_comparisons(rows)[0]
        self.assertEqual(row['referenceStepMs'],12)
        self.assertEqual(row['increasePercent'],25)
        self.assertTrue(row['identicalInputs'])
        self.assertFalse(row['identicalGeneratedOutputs'])

    def test_empty_union(self):
        self.assertEqual(union_length([]),0)

    def test_overlapping_and_nested_intervals(self):
        self.assertEqual(union_length([(0,10),(1,2),(5,15),(20,30)]),25)

    def test_touching_intervals(self):
        self.assertEqual(union_length([(0,1),(1,2)]),2)

    def test_negative_interval_rejected(self):
        with self.assertRaises(ValueError):
            union_length([(2,1)])

    def test_nonfinite_stat_rejected(self):
        for values in ([],[float('nan')],[float('inf')]):
            with self.assertRaises(ValueError):
                stats(values)

    def test_aggregate_runs_not_steps(self):
        self.assertEqual(stats([1,2,6]),dict(mean=3,min=1,max=6,count=3,p50=2))

    def test_group_priority(self):
        self.assertEqual(kernel_group('flashinfer::mnnvl::allreduce'),'collectives')
        self.assertEqual(kernel_group('bmm_MxE4m3_MxE2m1_MMA'),'moe_expert_gemm')
        self.assertEqual(kernel_group('paged_mqa_logits'),'index_scan_selection')
        self.assertEqual(kernel_group('sm100_mega_mhc'),'mhc')
        self.assertEqual(kernel_group('sm100_tf32_hc_prenorm_gemm_impl'),'mhc')
        self.assertEqual(kernel_group('cutlass_quantization_quantize'),'norm_position_quant')
        self.assertEqual(kernel_group('cutlass::blockscaled_gemm'),'other_gemm')

    def test_profile_union_not_nested_annotation_sum(self):
        events=[dict(cat='gpu_user_annotation',name='execute_1_context_0(x)_generation_1(x)',ts=0,dur=10,args={'External id':1}),
                dict(cat='gpu_user_annotation',name='execute_1_context_0(x)_generation_1(x)',ts=1,dur=8,args={'External id':1}),
                dict(cat='gpu_user_annotation',name='nested',ts=0,dur=10),
                dict(cat='kernel',name='gemm',ts=0,dur=6),
                dict(cat='kernel',name='allreduce',ts=3,dur=5)]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trace_rank0.123.gz'
            with gzip.open(path,'wt') as out:
                json.dump({'traceEvents':events},out)
            row=profile_summary(path)
        self.assertEqual(row['decodeSteps'],1)
        self.assertAlmostEqual(row['kernelDurationSumMs'],.011)
        self.assertAlmostEqual(row['activityUnionMs'],.008)
        self.assertAlmostEqual(row['noRecordedDeviceActivityMs'],.002)

    def test_missing_decode_annotation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trace_rank0.123.gz'
            with gzip.open(path,'wt') as out:
                json.dump({'traceEvents':[]},out)
            with self.assertRaises(ValueError):
                profile_summary(path)


if __name__=='__main__':
    unittest.main()
