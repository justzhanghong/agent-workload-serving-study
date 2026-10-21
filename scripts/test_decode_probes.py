#!/usr/bin/env python3
"""CPU-only tests of probe accounting; these do not validate GPU kernels."""
import io
import json
import threading
import unittest
from unittest.mock import patch
import decode_probe_client as client
from decode_attention_microbench import percentile
from decode_pd_client import route, score, common_prefix_length


class Response(io.BytesIO):
    status = 200


def events():
    data = [
        {'choices': [{'logprobs': {'tokens': ['token_id:123']}}]},
        {'choices': [{'logprobs': {'tokens': ['token_id:456']}}]},
        {'choices': [], 'usage': {'prompt_tokens': 17, 'completion_tokens': 2,
                                 'prompt_tokens_details': {'cached_tokens': 16}}},
    ]
    return b''.join(b'data: ' + json.dumps(d).encode() + b'\n\n' for d in data) + b'data: [DONE]\n\n'


class Tests(unittest.TestCase):
    def test_fixed_token_control_forwarded(self):
        with patch.object(client,'stream_request',return_value={}) as request:
            client.parallel('http://127.0.0.1','model',[[1,2]],256,'salt','timed',
                            extra={'allowed_token_ids':[123]})
        self.assertEqual(request.call_args.args[-1],{'allowed_token_ids':[123]})

    def test_empty_control_response(self):
        with patch.object(client.urllib.request, 'urlopen', return_value=Response(b'')):
            self.assertEqual(client.request_json('http://127.0.0.1/reset', {}),
                             {'http_status': 200, 'empty_response': True})

    def test_cache_reset_false_not_success(self):
        with patch.object(client,'request_json',return_value={'success':False}):
            with self.assertRaises(RuntimeError):
                client.reset_prefix_cache('http://127.0.0.1',wait_seconds=0)
        with patch.object(client,'request_json',return_value={'success':True}):
            self.assertEqual(client.reset_prefix_cache('http://127.0.0.1'),{'success':True})

    def test_cache_reset_unknown_schema_rejected(self):
        with patch.object(client,'request_json',return_value={'http_status':200,'empty_response':True}):
            with self.assertRaises(RuntimeError):
                client.reset_prefix_cache('http://127.0.0.1')

    def test_stream_accounting(self):
        with patch.object(client.urllib.request, 'urlopen', return_value=Response(events())):
            row = client.stream_request('http://127.0.0.1', 'model', list(range(17)), 2, 'test', 'req')
        self.assertIsNone(row['error'])
        self.assertEqual(row['output_token_ids'], [123, 456])
        self.assertEqual(row['cached_tokens'], 16)
        self.assertEqual(row['output_tokens'], 2)
        self.assertEqual(len(row['single_token_itl_ms']), 1)
        self.assertGreaterEqual(row['ttft_ms'], 0)

    def test_output_mismatch_not_success(self):
        with patch.object(client.urllib.request, 'urlopen', return_value=Response(events())):
            row = client.stream_request('http://127.0.0.1', 'model', [1], 3, 'test', 'req')
        with self.assertRaises(RuntimeError):
            client.ensure_success([row])

    def test_intentional_background_abort_is_labelled(self):
        stop = threading.Event()
        stop.set()
        with patch.object(client.urllib.request, 'urlopen', return_value=Response(events())):
            row = client.stream_request('http://127.0.0.1', 'model', [1], 8192, 'test', 'req', stop=stop)
        self.assertTrue(row['intentional_abort'])
        self.assertEqual(row['output_tokens'], 0)
        self.assertIsNone(row['tpot_ms'])

    def test_deterministic_inputs_and_percentiles(self):
        self.assertEqual(client.prompt(10, 123), client.prompt(10, 123))
        self.assertNotEqual(client.prompt(10, 123), client.prompt(10, 124))
        self.assertEqual(percentile([1, 2, 3], .5), 2)

    def test_ppd_score_and_weight_tradeoff(self):
        entry = dict(split='calibration', cell='cell1', history_length=8192,
                     append_length=128, output_budget=256, request_rate=1,
                     pd_ttft_ms=100, local_ttft_ms=50, pd_tpot_ms=10, local_tpot_ms=12)
        config = dict(mode='dynamic', request_rate=1, lookup_table=[entry])
        self.assertAlmostEqual(score(entry), .3)
        self.assertEqual(route(config, 1, 8192, 128, 256)[0], 'local')
        self.assertEqual(route(dict(config, tpot_weight=6), 1, 8192, 128, 256)[0], 'pd')

    def test_first_turn_always_pd(self):
        self.assertEqual(route(dict(mode='local'), 0, 0, 8192, 256), ('pd', None))

    def test_dynamic_rejects_evaluation_table(self):
        with self.assertRaises(ValueError):
            route(dict(mode='dynamic', lookup_table=[dict(split='evaluation')]), 1, 8192, 128, 256)

    def test_branch_and_compaction_use_actual_prefix(self):
        self.assertEqual(common_prefix_length([1,2,3],[1,2,3,4]),3)
        self.assertEqual(common_prefix_length([1,2,3],[1,2,7,8]),2)
        self.assertEqual(common_prefix_length([1,2,3],[7,8]),0)
        self.assertEqual(common_prefix_length([],[1,2]),0)


if __name__ == '__main__':
    unittest.main()
