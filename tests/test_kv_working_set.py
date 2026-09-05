from __future__ import annotations

import unittest
from dataclasses import fields
from pathlib import Path

from TokenSim.block.block_manager import BlockManager
from TokenSim.config.cache_config import CacheConfig
from TokenSim.config.psla_config import LLMResult
from TokenSim.config.config import ParallelConfig
from TokenSim.errors import ConfigurationError
from TokenSim.config.constants import _GB
from TokenSim.kv_working_set.config import (
    IO_SIZE_128K,
    MediaReadConfig,
    WorkingSetConfig,
)
from TokenSim.kv_working_set.fetch import (
    decode_fetch_for_requests,
    fetch_cost,
    layer_prefetch_step_latency,
    media_access_latency,
    media_n_ios,
    media_queue_latency,
    media_read_latency,
    qd_latency_us,
    spill_cost,
    spill_cost_for_requests,
)
from TokenSim.kv_working_set.placement import (
    attention_tokens,
    gpu_resident_blocks,
    gpu_resident_tokens,
    retained_tokens,
    split_context,
)
from TokenSim.kv_working_set.stats import WorkingSetStats
from TokenSim.latency import LLMCompassLatencyBackend, RooflineLatencyBackend
from TokenSim.latency.base import DECODE_SCALE
from TokenSim.llm.llm_request import Request
from TransformerRoofline import TransformerRoofline


REPO_ROOT = Path(__file__).resolve().parents[1]
SIZE_PER_TOKEN = 1024
SIZE_70B_TOKEN = 64 * 128 * 2 * 2 * 80
HIER_FRACS = (0.3, 0.5, 0.2)
SHIPPED_KV_WS = REPO_ROOT / "data" / "kv_working_set"


class _ModelStub:
    Nlayer = 80


class _RooflineStub:
    models = {"model": _ModelStub()}

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

    def test_compute_overlap_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            WorkingSetConfig(
                enabled=True,
                gpu_frac=1.0,
                overlap="compute_overlap",
            )

    def test_layer_prefetch_is_accepted(self):
        cfg = WorkingSetConfig(
            enabled=True,
            gpu_frac=1.0,
            overlap="layer_prefetch",
        )
        self.assertEqual(cfg.overlap, "layer_prefetch")

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
        self.assertEqual(hier.ssd.read_latency_us, 13.0)
        self.assertEqual(hier.ssd.read_bw_gbps, 14.0)
        self.assertEqual(hier.ssd.io_size_bytes, IO_SIZE_128K)
        self.assertEqual(hier.ssd.qd_cap, 32)
        self.assertEqual(hier.overlap, "layer_prefetch")
        self.assertEqual(hier.pcie_bw_gbps, 50.0)

    def test_negative_io_size_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            WorkingSetConfig(
                enabled=True,
                gpu_frac=0.5,
                dram_frac=0.5,
                ssd_frac=0.0,
                dram=MediaReadConfig(
                    read_latency_us=2.0,
                    read_bw_gbps=50.0,
                    io_size_bytes=-1,
                ),
            )

    def test_negative_sink_or_window_is_rejected(self):
        with self.assertRaises(ConfigurationError):
            WorkingSetConfig(enabled=True, gpu_frac=1.0, sink_tokens=-1)
        with self.assertRaises(ConfigurationError):
            WorkingSetConfig(enabled=True, gpu_frac=1.0, window_tokens=-1)

    def test_qd_cap_must_be_at_least_one(self):
        with self.assertRaises(ConfigurationError):
            WorkingSetConfig(
                enabled=True,
                gpu_frac=0.5,
                dram_frac=0.5,
                ssd_frac=0.0,
                dram=MediaReadConfig(
                    read_latency_us=2.0,
                    read_bw_gbps=50.0,
                    qd_cap=0,
                ),
            )

    def test_drive_presets_load(self):
        slc = WorkingSetConfig.from_file(SHIPPED_KV_WS / "hier_n3x_slc.json")
        mlc = WorkingSetConfig.from_file(SHIPPED_KV_WS / "hier_n3x.json")
        n3 = WorkingSetConfig.from_file(SHIPPED_KV_WS / "hier_n3.json")
        shipped = WorkingSetConfig.from_file(SHIPPED_KV_WS / "hier_30_50_20.json")
        self.assertEqual(slc.ssd.read_latency_us, 13.0)
        self.assertEqual(mlc.ssd.read_latency_us, 18.0)
        self.assertEqual(n3.ssd.read_latency_us, 50.0)
        for cfg in (slc, mlc, n3, shipped):
            self.assertEqual(cfg.ssd.read_bw_gbps, 14.0)
            self.assertEqual(cfg.ssd.io_size_bytes, IO_SIZE_128K)
            self.assertEqual(cfg.ssd.qd_cap, 32)
            self.assertEqual(cfg.overlap, "layer_prefetch")
            self.assertFalse(cfg.dram.queueing_enabled())
        self.assertEqual(shipped.ssd.read_latency_us, slc.ssd.read_latency_us)

    def test_shipped_4k_files_load(self):
        full = WorkingSetConfig.from_file(SHIPPED_KV_WS / "hier_30_50_20_slc_4k.json")
        sparse = WorkingSetConfig.from_file(
            SHIPPED_KV_WS / "hier_30_50_20_slc_4k_sparse256.json"
        )
        self.assertFalse(full.sparse)
        self.assertEqual(full.ssd.io_size_bytes, 4096)
        self.assertEqual(full.ssd.qd_cap, 64)
        self.assertEqual(full.ssd.qd_latency_us, ())
        self.assertTrue(sparse.sparse)
        self.assertEqual(sparse.sink_tokens, 4)
        self.assertEqual(sparse.window_tokens, 256)
        self.assertEqual(sparse.ssd.io_size_bytes, 4096)
        self.assertEqual(sparse.ssd.qd_cap, 64)
        streaming = WorkingSetConfig.from_file(
            SHIPPED_KV_WS / "streaming_sink4_window256.json"
        )
        self.assertTrue(streaming.streaming_attention)
        self.assertFalse(streaming.sparse)
        self.assertEqual(streaming.gpu_frac, 1.0)
        self.assertEqual(streaming.dram_frac, 0.0)
        self.assertEqual(streaming.ssd_frac, 0.0)
        self.assertEqual(streaming.sink_tokens, 4)
        self.assertEqual(streaming.window_tokens, 256)


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

    def test_every_decode_rereads_full_cold_set(self):
        config = _hier_config()
        first = fetch_cost(100, config, SIZE_PER_TOKEN)
        self.assertEqual((first.dram_tokens, first.ssd_tokens), (50, 20))
        second = fetch_cost(101, config, SIZE_PER_TOKEN)
        self.assertEqual((second.dram_tokens, second.ssd_tokens), (50, 21))
        self.assertGreater(second.latency, first.latency)

    def test_repeat_fetch_at_same_s_charges_again(self):
        config = _hier_config()
        first = fetch_cost(100, config, SIZE_PER_TOKEN)
        again = fetch_cost(100, config, SIZE_PER_TOKEN)
        self.assertAlmostEqual(again.latency, first.latency)
        self.assertEqual(again.dram_tokens, first.dram_tokens)
        self.assertEqual(again.ssd_tokens, first.ssd_tokens)

    def test_coalesced_io_size_matches_old_formula(self):
        config = _hier_config()
        cost = fetch_cost(100, config, SIZE_PER_TOKEN)
        dram_bytes = 50 * SIZE_PER_TOKEN
        ssd_bytes = 20 * SIZE_PER_TOKEN
        expected = (
            100.0 / 1e6
            + dram_bytes / _GB / 50.0
            + 100.0 / 1e6
            + ssd_bytes / _GB / 7.0
        )
        self.assertEqual(config.dram.io_size_bytes, 0)
        self.assertAlmostEqual(cost.latency, expected)
        self.assertAlmostEqual(
            media_access_latency(ssd_bytes, config.ssd),
            media_read_latency(ssd_bytes, config.ssd),
        )


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

    def test_second_backend_step_rereads_full_cold_set(self):
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
        ssd_after_first = hierarchical.kv_ws_stats.kv_ws_ssd_read_tokens
        request.generation_idx = 2
        baseline = disabled.estimate_step_latency(
            [_decode_request(prefill_len=99, generation_idx=2)]
        )
        observed = hierarchical.estimate_step_latency([request])
        full = fetch_cost(101, config, SIZE_PER_TOKEN)
        self.assertEqual((full.dram_tokens, full.ssd_tokens), (50, 21))
        self.assertAlmostEqual(observed, baseline + full.latency)
        self.assertEqual(
            hierarchical.kv_ws_stats.kv_ws_dram_read_tokens,
            dram_after_first + full.dram_tokens,
        )
        self.assertEqual(
            hierarchical.kv_ws_stats.kv_ws_ssd_read_tokens,
            ssd_after_first + full.ssd_tokens,
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

    def test_layer_prefetch_hides_fetch_when_compute_bound(self):
        request = _decode_request(prefill_len=99, generation_idx=1)
        config = WorkingSetConfig(
            enabled=True,
            placement="sliding_window",
            gpu_frac=HIER_FRACS[0],
            dram_frac=HIER_FRACS[1],
            ssd_frac=HIER_FRACS[2],
            overlap="layer_prefetch",
            dram=MediaReadConfig(read_latency_us=100.0, read_bw_gbps=50.0),
            ssd=MediaReadConfig(read_latency_us=100.0, read_bw_gbps=7.0),
        )
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
        compute = disabled.estimate_step_latency([request])
        fetch = fetch_cost(100, config, SIZE_PER_TOKEN).latency
        observed = hierarchical.estimate_step_latency([request])
        expected = layer_prefetch_step_latency(compute, fetch, 80)
        self.assertAlmostEqual(observed, expected)
        self.assertLess(observed, compute + fetch)
        self.assertAlmostEqual(
            hierarchical.kv_ws_stats.kv_ws_fetch_latency,
            fetch,
        )

    def test_layer_prefetch_stays_fetch_bound_when_io_dominates(self):
        compute = 0.01
        fetch = 2.0
        observed = layer_prefetch_step_latency(compute, fetch, 80)
        self.assertAlmostEqual(observed, fetch / 80 + fetch * 79 / 80)
        self.assertAlmostEqual(observed, fetch)


def _queued_ssd(latency_us: float, qd_cap: int = 32) -> MediaReadConfig:
    return MediaReadConfig(
        read_latency_us=latency_us,
        read_bw_gbps=14.0,
        io_size_bytes=4096,
        qd_cap=qd_cap,
    )


def _ssd_only_config(ssd: MediaReadConfig) -> WorkingSetConfig:
    return WorkingSetConfig(
        enabled=True,
        placement="sliding_window",
        gpu_frac=0.0,
        dram_frac=0.0,
        ssd_frac=1.0,
        overlap="blocking",
        ssd=ssd,
    )


class WorkingSetQueueingTest(unittest.TestCase):
    def test_70b_token_is_640_ios_at_4k(self):
        self.assertEqual(SIZE_70B_TOKEN, 2_621_440)
        media = _queued_ssd(13.0)
        self.assertEqual(media_n_ios(SIZE_70B_TOKEN, media), 640)

    def test_70b_token_is_20_ios_at_128k(self):
        media = MediaReadConfig(
            read_latency_us=13.0,
            read_bw_gbps=14.0,
            io_size_bytes=IO_SIZE_128K,
            qd_cap=32,
        )
        self.assertEqual(media_n_ios(SIZE_70B_TOKEN, media), 20)

    def test_qd_cap_eight_slower_than_512(self):
        bytes_ = SIZE_70B_TOKEN
        n_ios = 640
        slow = media_queue_latency(bytes_, n_ios, _queued_ssd(13.0, qd_cap=8))
        fast = media_queue_latency(bytes_, n_ios, _queued_ssd(13.0, qd_cap=512))
        self.assertGreater(slow, fast)

    def test_qd_cap_eight_slower_than_64(self):
        bytes_ = SIZE_70B_TOKEN
        n_ios = 640
        slow = media_queue_latency(bytes_, n_ios, _queued_ssd(13.0, qd_cap=8))
        fast = media_queue_latency(bytes_, n_ios, _queued_ssd(13.0, qd_cap=64))
        self.assertGreater(slow, fast)

    def test_no_table_qd64_13us_hits_bw_not_2p46m_iops(self):
        media = _queued_ssd(13.0, qd_cap=64)
        self.assertAlmostEqual(qd_latency_us(media, 64), 13.0)
        bytes_ = 820 * 327_680
        n_ios = media_n_ios(bytes_, media)
        self.assertEqual(n_ios, 65600)
        t = media_queue_latency(bytes_, n_ios, media)
        t_bw = bytes_ / _GB / 14.0
        t_iops_flat = n_ios * 13e-6 / 64
        t_iops_inflated = n_ios * 26e-6 / 64
        self.assertAlmostEqual(t, max(t_iops_flat, t_bw))
        self.assertAlmostEqual(t, t_bw)
        iops = n_ios / t
        self.assertGreater(iops, 3.0e6)
        self.assertLess(t, t_iops_inflated)

    def test_no_table_qd64_50us_is_slower_than_13us(self):
        bytes_ = 820 * 327_680
        n_ios = media_n_ios(bytes_, _queued_ssd(13.0, qd_cap=64))
        t13 = media_queue_latency(bytes_, n_ios, _queued_ssd(13.0, qd_cap=64))
        t50 = media_queue_latency(bytes_, n_ios, _queued_ssd(50.0, qd_cap=64))
        self.assertGreater(t50, t13)
        self.assertAlmostEqual(t50, n_ios * 50e-6 / 64)

    def test_qd_latency_table_still_interpolates(self):
        media = MediaReadConfig(
            read_latency_us=13.0,
            read_bw_gbps=14.0,
            io_size_bytes=4096,
            qd_cap=64,
            qd_latency_us=((1.0, 13.0), (32.0, 13.0), (64.0, 26.0)),
        )
        self.assertAlmostEqual(qd_latency_us(media, 32), 13.0)
        self.assertAlmostEqual(qd_latency_us(media, 64), 26.0)
        bytes_ = 820 * 327_680
        n_ios = media_n_ios(bytes_, media)
        t = media_queue_latency(bytes_, n_ios, media)
        t_iops = n_ios * 26e-6 / 64
        t_bw = bytes_ / _GB / 14.0
        self.assertAlmostEqual(t, max(t_iops, t_bw))

    def test_drive_latency_monotonic_at_cap_32(self):
        bytes_ = 200 * SIZE_70B_TOKEN
        n_ios = media_n_ios(bytes_, _queued_ssd(13.0))
        t13 = media_queue_latency(bytes_, n_ios, _queued_ssd(13.0))
        t18 = media_queue_latency(bytes_, n_ios, _queued_ssd(18.0))
        t50 = media_queue_latency(bytes_, n_ios, _queued_ssd(50.0))
        self.assertLess(t13, t18)
        self.assertLess(t18, t50)

    def test_batch_shares_ssd_queue(self):
        config = _ssd_only_config(_queued_ssd(13.0, qd_cap=32))
        reqs = [
            _LatencyRequest(prefill_len=9, generation_idx=1, is_prefill=False)
            for _ in range(2)
        ]
        cost = decode_fetch_for_requests(reqs, config, 4096)
        self.assertEqual(cost.ssd_tokens, 20)
        self.assertEqual(cost.ssd_ios, 20)
        shared = media_queue_latency(cost.ssd_bytes, cost.ssd_ios, config.ssd)
        self.assertAlmostEqual(cost.latency, shared)
        one = fetch_cost(10, config, 4096).latency
        self.assertAlmostEqual(shared, one)
        self.assertLess(shared, 2.0 * one - 1e-12)

    def test_queued_prefill_still_skips_fetch(self):
        request = _LatencyRequest(
            prefill_len=100,
            generation_idx=0,
            is_prefill=True,
        )
        config = _ssd_only_config(_queued_ssd(13.0))
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
            size_per_token=SIZE_70B_TOKEN,
        )
        self.assertAlmostEqual(
            hierarchical.estimate_step_latency([request]),
            disabled.estimate_step_latency([request]),
        )
        self.assertEqual(hierarchical.kv_ws_stats.kv_ws_fetch_latency, 0.0)
        self.assertEqual(hierarchical.kv_ws_stats.kv_ws_ssd_ios, 0)


class WorkingSetGpuOccupancyTest(unittest.TestCase):
    def test_resident_blocks_match_floor_frac(self):
        self.assertEqual(gpu_resident_tokens(1024, 0.3), 307)
        self.assertEqual(gpu_resident_blocks(1024, 16, 0.3), 20)
        self.assertEqual(gpu_resident_blocks(1024, 16, 1.0), 64)
        self.assertEqual(gpu_resident_blocks(512, 16, 0.1), 4)

    def test_prefill_allocates_full_context_regardless_of_gpu_frac(self):
        half = BlockManager(
            block_size=16, num_gpu_blocks=8, num_cpu_blocks=8, gpu_frac=0.5, watermark=0
        )
        req = Request(id=1, prefill_len=32, decode_len=16, block_size=16)
        self.assertTrue(half.can_allocate(req))
        half.allocate(req)
        self.assertEqual(half.block_table.get_num_blocks(1), 2)

    def test_trim_after_prefill_keeps_newest_gpu_frac_blocks(self):
        mgr = BlockManager(
            block_size=16, num_gpu_blocks=8, num_cpu_blocks=8, gpu_frac=0.5, watermark=0
        )
        req = Request(id=1, prefill_len=64, decode_len=16, block_size=16)
        mgr.allocate(req)
        self.assertEqual(mgr.block_table.get_num_blocks(1), 4)
        req.generation_idx = 1
        spilled = mgr.trim_to_gpu_target(req)
        target = gpu_resident_blocks(req.context_len, 16, 0.5)
        self.assertEqual(target, 2)
        self.assertEqual(spilled, 2)
        self.assertEqual(mgr.block_table.get_num_blocks(1), 2)

    def test_decode_gpu_frac_admits_more_after_trim(self):
        half = BlockManager(
            block_size=16, num_gpu_blocks=4, num_cpu_blocks=8, gpu_frac=0.5, watermark=0
        )
        fitted = []
        for req_id in range(10):
            req = Request(id=req_id, prefill_len=32, decode_len=1, block_size=16)
            if not half.can_allocate(req):
                break
            half.allocate(req)
            req.generation_idx = 1
            half.trim_to_gpu_target(req)
            fitted.append(req)
        self.assertEqual(len(fitted), 3)
        self.assertEqual(half.block_table.get_num_blocks(0), 1)


class WorkingSetSpillWriteTest(unittest.TestCase):
    def test_gpu_frac_one_spill_is_zero(self):
        cost = spill_cost(512, _hbm_only_config(), SIZE_PER_TOKEN)
        self.assertEqual(cost.latency, 0.0)
        self.assertEqual(cost.dram_bytes, 0)
        self.assertEqual(cost.ssd_bytes, 0)

    def test_spill_matches_cold_set_and_pcie_max(self):
        config = _hier_config()
        cost = spill_cost(100, config, SIZE_PER_TOKEN)
        self.assertEqual((cost.dram_tokens, cost.ssd_tokens), (50, 20))
        t_dram = media_access_latency(cost.dram_bytes, config.dram, write=True)
        t_ssd = media_access_latency(cost.ssd_bytes, config.ssd, write=True)
        t_pcie = (cost.dram_bytes + cost.ssd_bytes) / _GB / config.pcie_bw_gbps
        self.assertAlmostEqual(cost.latency, max(t_dram, t_ssd, t_pcie))

    def test_narrow_pcie_raises_spill(self):
        wide = _hier_config()
        narrow = WorkingSetConfig(
            enabled=True,
            placement="sliding_window",
            gpu_frac=HIER_FRACS[0],
            dram_frac=HIER_FRACS[1],
            ssd_frac=HIER_FRACS[2],
            overlap="blocking",
            pcie_bw_gbps=0.2,
            dram=MediaReadConfig(read_latency_us=100.0, read_bw_gbps=50.0),
            ssd=MediaReadConfig(read_latency_us=100.0, read_bw_gbps=7.0),
        )
        self.assertGreater(
            spill_cost(100, narrow, SIZE_PER_TOKEN).latency,
            spill_cost(100, wide, SIZE_PER_TOKEN).latency,
        )

    def test_prefill_request_with_decode_pays_spill(self):
        config = _hier_config()
        req = _LatencyRequest(prefill_len=99, generation_idx=0, is_prefill=True)
        req.decode_len = 16
        cost = spill_cost_for_requests([req], config, SIZE_PER_TOKEN)
        expected = spill_cost(100, config, SIZE_PER_TOKEN)
        self.assertAlmostEqual(cost.latency, expected.latency)

    def test_prefill_without_remaining_decode_skips_spill(self):
        config = _hier_config()
        req = _LatencyRequest(prefill_len=99, generation_idx=0, is_prefill=True)
        req.decode_len = 1
        cost = spill_cost_for_requests([req], config, SIZE_PER_TOKEN)
        self.assertEqual(cost.latency, 0.0)

    def test_spill_fields_are_on_llmresult_and_stats_dict(self):
        names = {field.name for field in fields(LLMResult)}
        for key in (
            "kv_ws_spill_latency",
            "kv_ws_dram_write_bytes",
            "kv_ws_ssd_write_bytes",
        ):
            self.assertIn(key, names)
        cost = spill_cost(100, _hier_config(), SIZE_PER_TOKEN)
        stats = WorkingSetStats.from_config(_hier_config())
        stats.record_spill(cost)
        payload = stats.as_dict()
        self.assertAlmostEqual(payload["kv_ws_spill_latency"], cost.latency)
        self.assertEqual(payload["kv_ws_dram_write_bytes"], cost.dram_bytes)
        self.assertEqual(payload["kv_ws_ssd_write_bytes"], cost.ssd_bytes)


def _sparse_config(
    gpu_frac: float = HIER_FRACS[0],
    dram_frac: float = HIER_FRACS[1],
    ssd_frac: float = HIER_FRACS[2],
    sink_tokens: int = 4,
    window_tokens: int = 256,
) -> WorkingSetConfig:
    return WorkingSetConfig(
        enabled=True,
        placement="sliding_window",
        gpu_frac=gpu_frac,
        dram_frac=dram_frac,
        ssd_frac=ssd_frac,
        overlap="blocking",
        sparse=True,
        sink_tokens=sink_tokens,
        window_tokens=window_tokens,
        dram=MediaReadConfig(read_latency_us=2.0, read_bw_gbps=50.0),
        ssd=MediaReadConfig(read_latency_us=13.0, read_bw_gbps=14.0),
    )


class WorkingSetSparseFetchTest(unittest.TestCase):
    def test_sparse_false_matches_full_split_and_fetch(self):
        full = _hier_config()
        tagged = WorkingSetConfig(
            enabled=True,
            placement="sliding_window",
            gpu_frac=HIER_FRACS[0],
            dram_frac=HIER_FRACS[1],
            ssd_frac=HIER_FRACS[2],
            overlap="blocking",
            sparse=False,
            sink_tokens=4,
            window_tokens=256,
            dram=MediaReadConfig(read_latency_us=100.0, read_bw_gbps=50.0),
            ssd=MediaReadConfig(read_latency_us=100.0, read_bw_gbps=7.0),
        )
        for s in (2048, 4096):
            self.assertEqual(split_context(s, tagged), split_context(s, full))
            a = fetch_cost(s, tagged, SIZE_PER_TOKEN)
            b = fetch_cost(s, full, SIZE_PER_TOKEN)
            self.assertEqual((a.dram_tokens, a.ssd_tokens), (b.dram_tokens, b.ssd_tokens))
            self.assertAlmostEqual(a.latency, b.latency)

    def test_s4096_full_vs_sparse_split_and_io(self):
        full = split_context(4096, _hier_config())
        self.assertEqual((full.gpu, full.dram, full.ssd), (1228, 2048, 820))
        sparse = split_context(4096, _sparse_config())
        self.assertEqual((sparse.gpu, sparse.dram, sparse.ssd), (1232, 2048, 816))
        self.assertEqual(gpu_resident_tokens(4096, 0.3), 1228)
        self.assertEqual(gpu_resident_tokens(4096, 0.3, 4), 1232)
        self.assertEqual(gpu_resident_blocks(4096, 16, 0.3), 77)
        self.assertEqual(gpu_resident_blocks(4096, 16, 0.3, 4), 77)
        cost = fetch_cost(4096, _sparse_config(), 327_680)
        self.assertEqual((cost.dram_tokens, cost.ssd_tokens), (0, 0))
        self.assertEqual(cost.ssd_ios, 0)
        spill = spill_cost(4096, _sparse_config(), 327_680)
        self.assertEqual(spill.ssd_tokens, 816)
        self.assertEqual(spill.dram_tokens, 2048)

    def test_s2048_full_ssd_is_410_sparse_fetch_still_zero(self):
        full = split_context(2048, _hier_config())
        self.assertEqual((full.gpu, full.dram, full.ssd), (614, 1024, 410))
        self.assertLess(full.ssd, 820)
        sparse = fetch_cost(2048, _sparse_config(), 327_680)
        self.assertEqual((sparse.dram_tokens, sparse.ssd_tokens), (0, 0))
        end = fetch_cost(4096, _sparse_config(), 327_680)
        self.assertEqual((end.dram_tokens, end.ssd_tokens), (0, 0))

    def test_s1024_gpu10_window256_reads_dram_only(self):
        config = _sparse_config(gpu_frac=0.1, dram_frac=0.5, ssd_frac=0.4)
        split = split_context(1024, config)
        self.assertEqual(split.gpu, 106)
        cost = fetch_cost(1024, config, SIZE_PER_TOKEN)
        self.assertEqual((cost.dram_tokens, cost.ssd_tokens), (154, 0))

    def test_short_context_reads_full_cold_set(self):
        config = _sparse_config()
        split = split_context(100, config)
        self.assertEqual((split.gpu, split.dram, split.ssd), (34, 50, 16))
        cost = fetch_cost(100, config, SIZE_PER_TOKEN)
        self.assertEqual((cost.dram_tokens, cost.ssd_tokens), (50, 16))

    def test_s4096_gpu30_dram0_ssd70_full_reads_ssd_sparse_does_not(self):
        full = WorkingSetConfig(
            enabled=True,
            placement="sliding_window",
            gpu_frac=0.3,
            dram_frac=0.0,
            ssd_frac=0.7,
            overlap="blocking",
            ssd=MediaReadConfig(read_latency_us=13.0, read_bw_gbps=14.0),
        )
        sparse = _sparse_config(gpu_frac=0.3, dram_frac=0.0, ssd_frac=0.7)
        full_split = split_context(4096, full)
        sparse_split = split_context(4096, sparse)
        self.assertEqual((full_split.gpu, full_split.dram, full_split.ssd), (1228, 0, 2868))
        self.assertEqual(
            (sparse_split.gpu, sparse_split.dram, sparse_split.ssd), (1232, 0, 2864)
        )
        full_cost = fetch_cost(4096, full, 327_680)
        sparse_cost = fetch_cost(4096, sparse, 327_680)
        self.assertEqual((full_cost.dram_tokens, full_cost.ssd_tokens), (0, 2868))
        self.assertGreater(full_cost.ssd_ios, 0)
        self.assertEqual((sparse_cost.dram_tokens, sparse_cost.ssd_tokens), (0, 0))
        self.assertEqual(sparse_cost.ssd_ios, 0)
        spill = spill_cost(4096, sparse, 327_680)
        self.assertEqual((spill.dram_tokens, spill.ssd_tokens), (0, 2864))

    def test_sparse_keeps_middle_kv_and_attends_sink_window(self):
        config = _sparse_config()
        split = split_context(4096, config)
        self.assertEqual((split.gpu, split.dram, split.ssd), (1232, 2048, 816))
        self.assertEqual(attention_tokens(4096, config), 260)
        self.assertEqual(attention_tokens(4096, _hier_config()), 4096)

        class _RecordRoofline(_RooflineStub):
            def __init__(self):
                self.calls = []

            def Compute_Timebreakdown_Iteration(
                self,
                prefill_len,
                generation_idx,
                batch_size,
                model,
                hardware,
                Pipeline_Stage,
            ):
                self.calls.append((prefill_len, generation_idx, batch_size))
                return 0.01, 0.0001 * (prefill_len + generation_idx)

        rf = _RecordRoofline()
        backend = RooflineLatencyBackend(
            rf,
            "model",
            "hardware",
            ParallelConfig(),
            working_set_config=config,
        )
        backend.estimate_step_latency(
            [_decode_request(prefill_len=2048, generation_idx=2048)]
        )
        attn_calls = [c for c in rf.calls if c[2] == 1]
        self.assertTrue(attn_calls)
        prompt, step, _ = attn_calls[-1]
        self.assertEqual(prompt + step, 260)


def _streaming_config(
    sink_tokens: int = 4,
    window_tokens: int = 256,
) -> WorkingSetConfig:
    return WorkingSetConfig(
        enabled=True,
        placement="sliding_window",
        gpu_frac=1.0,
        dram_frac=0.0,
        ssd_frac=0.0,
        overlap="blocking",
        streaming_attention=True,
        sink_tokens=sink_tokens,
        window_tokens=window_tokens,
    )


class WorkingSetStreamingAttentionTest(unittest.TestCase):
    def test_retained_tokens_sink_window_no_double_count(self):
        self.assertEqual(retained_tokens(100, 4, 256), 100)
        self.assertEqual(retained_tokens(260, 4, 256), 260)
        self.assertEqual(retained_tokens(4096, 4, 256), 260)
        self.assertEqual(retained_tokens(4, 4, 256), 4)
        self.assertEqual(retained_tokens(0, 4, 256), 0)

    def test_streaming_split_evicts_middle(self):
        split = split_context(4096, _streaming_config())
        self.assertEqual((split.gpu, split.dram, split.ssd), (260, 0, 0))
        self.assertEqual(split.sink, 4)
        self.assertEqual(gpu_resident_tokens(4096, 1.0, 4, 256, True), 260)
        self.assertEqual(gpu_resident_blocks(4096, 16, 1.0, 4, 256, True), 17)

    def test_streaming_fetch_and_spill_are_zero(self):
        config = _streaming_config()
        cost = fetch_cost(4096, config, 327_680)
        self.assertEqual((cost.dram_tokens, cost.ssd_tokens), (0, 0))
        self.assertEqual(cost.latency, 0.0)
        spill = spill_cost(4096, config, 327_680)
        self.assertEqual(spill.latency, 0.0)
        self.assertEqual((spill.dram_bytes, spill.ssd_bytes), (0, 0))
        req = _LatencyRequest(prefill_len=2048, generation_idx=0, is_prefill=True)
        req.decode_len = 2048
        self.assertEqual(
            spill_cost_for_requests([req], config, 327_680).latency, 0.0
        )

    def test_trim_keeps_sink_and_window(self):
        mgr = BlockManager(
            block_size=16,
            num_gpu_blocks=256,
            num_cpu_blocks=8,
            gpu_frac=1.0,
            watermark=0,
            sink_tokens=4,
            window_tokens=256,
            streaming_attention=True,
        )
        req = Request(id=1, prefill_len=512, decode_len=16, block_size=16)
        mgr.allocate(req)
        self.assertEqual(mgr.block_table.get_num_blocks(1), 32)
        req.generation_idx = 1
        spilled = mgr.trim_to_gpu_target(req)
        self.assertEqual(mgr.block_table.get_num_blocks(1), 17)
        self.assertEqual(spilled, 15)
        kept = mgr.block_table.get_blocks(1)
        all_blocks = list(range(32))
        # First block is sink; last 16 are the window.
        self.assertEqual(kept[0].block_number, 0)
        self.assertEqual([b.block_number for b in kept[1:]], list(range(16, 32)))
        self.assertEqual(len(all_blocks) - 17, spilled)

    def test_decode_admits_more_after_streaming_trim(self):
        mgr = BlockManager(
            block_size=16,
            num_gpu_blocks=64,
            num_cpu_blocks=8,
            gpu_frac=1.0,
            watermark=0,
            sink_tokens=4,
            window_tokens=256,
            streaming_attention=True,
        )
        fitted = []
        for req_id in range(10):
            req = Request(id=req_id, prefill_len=384, decode_len=1, block_size=16)
            if not mgr.can_allocate(req):
                break
            mgr.allocate(req)
            req.generation_idx = 1
            mgr.trim_to_gpu_target(req)
            fitted.append(req)
        self.assertEqual(len(fitted), 3)
        self.assertEqual(mgr.block_table.get_num_blocks(0), 17)

    def test_decode_attention_uses_retained_context(self):
        class _RecordRoofline(_RooflineStub):
            def __init__(self):
                self.calls = []

            def Compute_Timebreakdown_Iteration(
                self,
                prefill_len,
                generation_idx,
                batch_size,
                model,
                hardware,
                Pipeline_Stage,
            ):
                self.calls.append((prefill_len, generation_idx, batch_size))
                return 0.01, 0.0001 * (prefill_len + generation_idx)

        full_rf = _RecordRoofline()
        sparse_rf = _RecordRoofline()
        request = _decode_request(prefill_len=2048, generation_idx=2048)
        full = RooflineLatencyBackend(full_rf, "model", "hardware", ParallelConfig())
        streaming = RooflineLatencyBackend(
            sparse_rf,
            "model",
            "hardware",
            ParallelConfig(),
            working_set_config=_streaming_config(),
        )
        full_lat = full.estimate_step_latency([request])
        stream_lat = streaming.estimate_step_latency([request])
        self.assertLess(stream_lat, full_lat)
        attn_calls = [c for c in sparse_rf.calls if c[2] == 1]
        self.assertTrue(attn_calls)
        prompt, step, _ = attn_calls[-1]
        self.assertEqual(prompt + step, 260)

    def test_prefill_attention_stays_full_length(self):
        class _RecordRoofline(_RooflineStub):
            def __init__(self):
                self.calls = []

            def Compute_Timebreakdown_Iteration(
                self,
                prefill_len,
                generation_idx,
                batch_size,
                model,
                hardware,
                Pipeline_Stage,
            ):
                self.calls.append((prefill_len, generation_idx, batch_size))
                return 0.01, 0.001

        rf = _RecordRoofline()
        request = _LatencyRequest(
            prefill_len=2048, generation_idx=0, is_prefill=True
        )
        backend = RooflineLatencyBackend(
            rf,
            "model",
            "hardware",
            ParallelConfig(),
            working_set_config=_streaming_config(),
        )
        disabled = RooflineLatencyBackend(
            _RooflineStub(), "model", "hardware", ParallelConfig()
        )
        self.assertAlmostEqual(
            backend.estimate_step_latency([request]),
            disabled.estimate_step_latency([request]),
        )


class WorkingSetGqaCacheTest(unittest.TestCase):
    def test_gqa_kv_is_one_eighth_of_mha(self):
        roofline = TransformerRoofline(
            str(REPO_ROOT / "TransformerRoofline/hardware_models.json"),
            str(REPO_ROOT / "TransformerRoofline/allreduce_v100.xlsx"),
            str(REPO_ROOT / "TransformerRoofline/hardware_elements.json"),
        )
        mha = CacheConfig(16, "H200", "LLaMa2-70B", roofline)
        gqa = CacheConfig(16, "H200", "LLaMa2-70B-GQA", roofline)
        self.assertEqual(mha.num_kv_heads, 64)
        self.assertEqual(gqa.num_kv_heads, 8)
        self.assertEqual(mha.size_per_token, 2_621_440)
        self.assertEqual(gqa.size_per_token, mha.size_per_token // 8)
        self.assertEqual(gqa.size_per_token, 327_680)
        self.assertEqual(int(mha.num_gpu_blocks), 518)
        self.assertEqual(int(gqa.num_gpu_blocks), 4144)


if __name__ == "__main__":
    unittest.main()
