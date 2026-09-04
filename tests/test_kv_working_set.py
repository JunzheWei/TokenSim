from __future__ import annotations

import unittest
from pathlib import Path

from TokenSim.config.config import ParallelConfig
from TokenSim.config.constants import _GB
from TokenSim.errors import ConfigurationError
from TokenSim.kv_working_set.config import MediaReadConfig, WorkingSetConfig
from TokenSim.kv_working_set.fetch import fetch_cost, media_read_latency
from TokenSim.kv_working_set.placement import split_context
from TokenSim.latency import LLMCompassLatencyBackend, RooflineLatencyBackend
from TokenSim.latency.base import DECODE_SCALE


REPO_ROOT = Path(__file__).resolve().parents[1]
SIZE_PER_TOKEN = 1024
HIER_FRACS = (0.3, 0.5, 0.2)


class _RooflineStub:
    def Compute_Timebreakdown_Iteration(
        self,
        prefill_len,
        generation_idx,
        batch_size,
        model,
        hardware,
        Pipeline_Stage,
    ):
        return 0.01, 0.001


class _LatencyRequest:
    def __init__(
        self,
        prefill_len: int,
        generation_idx: int,
        is_prefill: bool,
        needs_recompute: bool = False,
        recompute_tokens: int = 0,
    ):
        self.prefill_len = prefill_len
        self.generation_idx = generation_idx
        self.is_prefill = is_prefill
        self.kv_ws_fetched_end = 0
        self.prefill_compute_len = prefill_len
        self.needs_recompute = needs_recompute
        self.recompute_tokens = recompute_tokens


def _hier_config() -> WorkingSetConfig:
    return WorkingSetConfig(
        enabled=True,
        placement="sliding_window",
        gpu_frac=HIER_FRACS[0],
        dram_frac=HIER_FRACS[1],
        ssd_frac=HIER_FRACS[2],
        overlap="blocking",
        dram=MediaReadConfig(read_latency_us=100.0, read_bw_gbps=50.0),
        ssd=MediaReadConfig(read_latency_us=100.0, read_bw_gbps=7.0),
    )


def _hbm_only_config() -> WorkingSetConfig:
    return WorkingSetConfig(
        enabled=True,
        placement="sliding_window",
        gpu_frac=1.0,
        dram_frac=0.0,
        ssd_frac=0.0,
        overlap="blocking",
    )


def _decode_request(prefill_len: int = 99, generation_idx: int = 1) -> _LatencyRequest:
    return _LatencyRequest(
        prefill_len=prefill_len,
        generation_idx=generation_idx,
        is_prefill=False,
    )


class WorkingSetConfigTest(unittest.TestCase):
    def test_fractions_must_sum_to_one(self):
        with self.assertRaises(ConfigurationError):
            WorkingSetConfig(
                enabled=True,
                gpu_frac=0.5,
                dram_frac=0.5,
                ssd_frac=0.5,
            )

    def test_negative_bandwidth_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            WorkingSetConfig(
                enabled=True,
                gpu_frac=0.5,
                dram_frac=0.5,
                ssd_frac=0.0,
                dram=MediaReadConfig(read_latency_us=10.0, read_bw_gbps=-1.0),
            )

    def test_negative_latency_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            WorkingSetConfig(
                enabled=True,
                gpu_frac=0.5,
                dram_frac=0.5,
                ssd_frac=0.0,
                dram=MediaReadConfig(read_latency_us=-1.0, read_bw_gbps=10.0),
            )

    def test_compute_overlap_is_rejected_in_v1(self):
        with self.assertRaises(ConfigurationError):
            WorkingSetConfig(
                enabled=True,
                gpu_frac=1.0,
                overlap="compute_overlap",
            )

    def test_missing_media_when_frac_positive(self):
        with self.assertRaises(ConfigurationError):
            WorkingSetConfig(
                enabled=True,
                gpu_frac=0.5,
                dram_frac=0.5,
                ssd_frac=0.0,
            )

    def test_shipped_example_files_load(self):
        hbm = WorkingSetConfig.from_file(
            REPO_ROOT / "data/kv_working_set/hbm_only.json"
        )
        hier = WorkingSetConfig.from_file(
            REPO_ROOT / "data/kv_working_set/hier_30_50_20.json"
        )
        self.assertTrue(hbm.enabled)
        self.assertEqual(hbm.gpu_frac, 1.0)
        self.assertEqual(hier.gpu_frac, 0.3)
        self.assertEqual(hier.dram_frac, 0.5)
        self.assertEqual(hier.ssd_frac, 0.2)


class WorkingSetPlacementTest(unittest.TestCase):
    def test_split_s100_30_50_20_newest_on_gpu(self):
        split = split_context(100, _hier_config())
        self.assertEqual(split.gpu + split.dram + split.ssd, 100)
        self.assertEqual((split.gpu, split.dram, split.ssd), (30, 50, 20))
        self.assertEqual(split.gpu_start, 70)
        self.assertEqual(split.dram_start, 20)
        self.assertEqual(split.ssd_start, 0)

    def test_zero_dram_and_ssd_tokens_fetch_is_zero(self):
        cost = fetch_cost(100, _hbm_only_config(), SIZE_PER_TOKEN)
        self.assertEqual(cost.dram_tokens, 0)
        self.assertEqual(cost.ssd_tokens, 0)
        self.assertEqual(cost.latency, 0.0)

    def test_second_decode_only_faults_new_storage_tokens(self):
        config = _hier_config()
        first = fetch_cost(100, config, SIZE_PER_TOKEN, fetched_end=0)
        self.assertEqual((first.dram_tokens, first.ssd_tokens), (50, 20))
        self.assertEqual(first.next_fetched_end, 70)
        second = fetch_cost(101, config, SIZE_PER_TOKEN, fetched_end=first.next_fetched_end)
        self.assertEqual(second.ssd_tokens, 0)
        self.assertEqual(second.dram_tokens, 1)
        self.assertEqual(second.next_fetched_end, 71)
        self.assertLess(second.latency, first.latency)

    def test_repeat_fetch_at_same_s_is_zero(self):
        config = _hier_config()
        first = fetch_cost(100, config, SIZE_PER_TOKEN)
        again = fetch_cost(100, config, SIZE_PER_TOKEN, fetched_end=first.next_fetched_end)
        self.assertEqual(again.latency, 0.0)
        self.assertEqual(again.dram_tokens, 0)
        self.assertEqual(again.ssd_tokens, 0)


class WorkingSetLatencyBackendTest(unittest.TestCase):
    def test_gpu_frac_one_matches_disabled_decode_latency(self):
        request = _decode_request()
        disabled = RooflineLatencyBackend(
            _RooflineStub(),
            "model",
            "hardware",
            ParallelConfig(),
        )
        hbm_only = RooflineLatencyBackend(
            _RooflineStub(),
            "model",
            "hardware",
            ParallelConfig(),
            working_set_config=_hbm_only_config(),
            size_per_token=SIZE_PER_TOKEN,
        )
        self.assertAlmostEqual(
            disabled.estimate_step_latency([request]),
            hbm_only.estimate_step_latency([request]),
        )

    def test_hierarchical_decode_adds_exact_fetch(self):
        request = _decode_request(prefill_len=99, generation_idx=1)
        config = _hier_config()
        disabled = RooflineLatencyBackend(
            _RooflineStub(),
            "model",
            "hardware",
            ParallelConfig(),
        )
        hierarchical = RooflineLatencyBackend(
            _RooflineStub(),
            "model",
            "hardware",
            ParallelConfig(),
            working_set_config=config,
            size_per_token=SIZE_PER_TOKEN,
        )
        baseline = disabled.estimate_step_latency([request])
        observed = hierarchical.estimate_step_latency([request])
        expected_fetch = fetch_cost(100, config, SIZE_PER_TOKEN).latency
        self.assertAlmostEqual(observed, baseline + expected_fetch)
        self.assertAlmostEqual(
            hierarchical.kv_ws_stats.kv_ws_fetch_latency,
            expected_fetch,
        )
        self.assertEqual(hierarchical.kv_ws_stats.kv_ws_dram_read_tokens, 50)
        self.assertEqual(hierarchical.kv_ws_stats.kv_ws_ssd_read_tokens, 20)

    def test_prefill_does_not_add_fetch(self):
        request = _LatencyRequest(
            prefill_len=100,
            generation_idx=0,
            is_prefill=True,
        )
        config = _hier_config()
        disabled = RooflineLatencyBackend(
            _RooflineStub(),
            "model",
            "hardware",
            ParallelConfig(),
        )
        hierarchical = RooflineLatencyBackend(
            _RooflineStub(),
            "model",
            "hardware",
            ParallelConfig(),
            working_set_config=config,
            size_per_token=SIZE_PER_TOKEN,
        )
        self.assertAlmostEqual(
            hierarchical.estimate_step_latency([request]),
            disabled.estimate_step_latency([request]),
        )
        self.assertEqual(hierarchical.kv_ws_stats.kv_ws_fetch_latency, 0.0)

    def test_recompute_does_not_add_fetch(self):
        request = _LatencyRequest(
            prefill_len=100,
            generation_idx=8,
            is_prefill=False,
            needs_recompute=True,
            recompute_tokens=108,
        )
        config = _hier_config()
        disabled = RooflineLatencyBackend(
            _RooflineStub(),
            "model",
            "hardware",
            ParallelConfig(),
        )
        hierarchical = RooflineLatencyBackend(
            _RooflineStub(),
            "model",
            "hardware",
            ParallelConfig(),
            working_set_config=config,
            size_per_token=SIZE_PER_TOKEN,
        )
        self.assertAlmostEqual(
            hierarchical.estimate_step_latency([request]),
            disabled.estimate_step_latency([request]),
        )
        self.assertEqual(hierarchical.kv_ws_stats.kv_ws_fetch_latency, 0.0)

    def test_llmcompass_decode_adds_same_fetch(self):
        request = _decode_request(prefill_len=99, generation_idx=1)
        config = _hier_config()
        fallback = RooflineLatencyBackend(
            _RooflineStub(),
            "model",
            "hardware",
            ParallelConfig(),
            working_set_config=config,
            size_per_token=SIZE_PER_TOKEN,
        )
        backend = LLMCompassLatencyBackend(
            fallback_backend=fallback,
            wrapped_llmcompass_vars=(None, None, None),
        )
        backend._estimate_llmcompass = lambda requests: 0.05  # noqa: ARG005
        observed = backend.estimate_step_latency([request])
        expected_fetch = fetch_cost(100, config, SIZE_PER_TOKEN).latency
        self.assertAlmostEqual(observed, 0.05 + expected_fetch)

    def test_second_backend_step_only_faults_window_slide(self):
        request = _decode_request(prefill_len=99, generation_idx=1)
        config = _hier_config()
        hierarchical = RooflineLatencyBackend(
            _RooflineStub(),
            "model",
            "hardware",
            ParallelConfig(),
            working_set_config=config,
            size_per_token=SIZE_PER_TOKEN,
        )
        disabled = RooflineLatencyBackend(
            _RooflineStub(),
            "model",
            "hardware",
            ParallelConfig(),
        )
        hierarchical.estimate_step_latency([request])
        dram_after_first = hierarchical.kv_ws_stats.kv_ws_dram_read_tokens
        request.generation_idx = 2
        baseline = disabled.estimate_step_latency(
            [_decode_request(prefill_len=99, generation_idx=2)]
        )
        observed = hierarchical.estimate_step_latency([request])
        slide = fetch_cost(101, config, SIZE_PER_TOKEN, fetched_end=70)
        self.assertEqual(slide.dram_tokens, 1)
        self.assertEqual(slide.ssd_tokens, 0)
        self.assertAlmostEqual(observed, baseline + slide.latency)
        self.assertEqual(
            hierarchical.kv_ws_stats.kv_ws_dram_read_tokens,
            dram_after_first + 1,
        )

    def test_decode_scale_is_not_applied_to_fetch(self):
        request = _decode_request(prefill_len=99, generation_idx=1)
        config = _hier_config()
        hierarchical = RooflineLatencyBackend(
            _RooflineStub(),
            "model",
            "hardware",
            ParallelConfig(),
            working_set_config=config,
            size_per_token=SIZE_PER_TOKEN,
        )
        fetch = fetch_cost(100, config, SIZE_PER_TOKEN).latency
        observed = hierarchical.estimate_step_latency([request])
        roofline_decode = observed - fetch
        self.assertAlmostEqual(roofline_decode, 0.011 * DECODE_SCALE)


if __name__ == "__main__":
    unittest.main()
