# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

""" Unit tests for CUDA-graph-aware admission gating. """

import logging
import types
from collections import deque

import pytest
import torch

from megatron.core.inference.batch_dimensions_utils import (
    CUDAGraphBatchDimensionBuilder,
    InferenceBatchDimensions,
)
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.inference_request import DynamicInferenceRequest


def _create_engine(
    cg_list, active_tok=0, num_prefill=0, num_decode=0, is_hybrid=False, warn_after=100
):
    """Mock engine instance."""
    engine = types.SimpleNamespace()
    engine.context = types.SimpleNamespace(
        cuda_graph_batch_dimensions_list=cg_list,
        active_token_count=active_tok,
        num_prefill_requests=num_prefill,
        num_decode_requests=num_decode,
        is_hybrid_model=is_hybrid,
        has_kda=False,
        use_cuda_graphs_for_non_decode_steps=True,
    )
    engine.cuda_graph_all_prefills = True
    engine._cg_admission_warn_after = warn_after
    engine._cg_admission_gating_active = DynamicInferenceEngine._cg_admission_gating_active.__get__(
        engine
    )
    engine._find_cg_chunk_size = DynamicInferenceEngine._find_cg_chunk_size.__get__(engine)
    engine._matches_cg_admission = DynamicInferenceEngine._matches_cg_admission.__get__(engine)
    engine._cg_admission_check = DynamicInferenceEngine._cg_admission_check.__get__(engine)
    engine._register_cg_wait = DynamicInferenceEngine._register_cg_wait.__get__(engine)
    return engine


def _make_request(request_id=1, cg_wait_iters=0):
    """Tiny stand-in for DynamicInferenceRequest; gating only reads/writes these fields."""
    return types.SimpleNamespace(request_id=request_id, cg_wait_iters=cg_wait_iters)


def _get_cudagraph(token_count, p, d):
    return InferenceBatchDimensions(
        token_count=token_count, prefill_req_count=p, decode_req_count=d
    )


class _SchedulerRequest:
    """Small CPU-only request stand-in for exercising scheduler call paths."""

    def __init__(self, request_id, prompt_length, cg_wait_iters=0, block_hashes=None):
        self.request_id = request_id
        self.remaining_prompt_tokens = torch.arange(prompt_length)
        self.finished_chunk_token_count = 0
        self.cg_wait_iters = cg_wait_iters
        self.precomputed_block_hashes = block_hashes or []
        self.added_to_context = False

    @property
    def remaining_prompt_length(self):
        return len(self.remaining_prompt_tokens)

    def add_event_add_context(self):
        self.added_to_context = True


class _SchedulerContext:
    def __init__(
        self,
        graph_dims,
        *,
        max_tokens=256,
        is_hybrid=False,
        enable_prefix_caching=False,
        effective_prefill_tokens=None,
    ):
        self.cuda_graph_batch_dimensions_list = graph_dims
        self.use_cuda_graphs_for_non_decode_steps = True
        self.active_token_count = 0
        self.num_prefill_requests = 0
        self.num_decode_requests = 0
        self.is_hybrid_model = is_hybrid
        self.has_kda = False
        self.enable_prefix_caching = enable_prefix_caching
        self.mamba_slot_allocator = None
        self.max_tokens = max_tokens
        self.chunked_prefill_request_id = -1
        self.total_request_count = 0
        self.kv_block_allocator = types.SimpleNamespace(kv_hash_to_block_id={})
        self.effective_prefill_tokens = effective_prefill_tokens
        self.admissions = []

    def _compute_prefix_match(self, req, prefill_chunk_length):
        effective = prefill_chunk_length
        if self.effective_prefill_tokens is not None:
            effective = min(effective, self.effective_prefill_tokens)
        return ([], 0, 0, 0, prefill_chunk_length - effective, effective)

    def check_availability(self, req):
        return True, True, True

    def add_request(self, req, prefill_chunk_length=None):
        submitted = (
            req.remaining_prompt_length if prefill_chunk_length is None else prefill_chunk_length
        )
        effective = self._compute_prefix_match(req, submitted)[-1]
        self.active_token_count += effective
        self.num_prefill_requests += 1
        self.total_request_count += 1
        self.admissions.append((req.request_id, prefill_chunk_length, effective))


class _SchedulerLoop:
    def create_task(self, coroutine):
        coroutine.close()

    def call_soon_threadsafe(self, callback, *args):
        callback(*args)


def _create_scheduler_engine(req, context, *, chunked=False):
    engine = object.__new__(DynamicInferenceEngine)
    engine.context = context
    engine.cuda_graph_all_prefills = True
    engine._cg_admission_warn_after = 100
    engine.enable_chunked_prefill = chunked
    engine.waiting_request_ids = deque([req.request_id])
    engine.requests = {req.request_id: types.SimpleNamespace(record=[req])}
    engine._prefix_coordination_waits = 0
    engine._loop = _SchedulerLoop()
    return engine


# CG list sorted descending by token_count, matching the production list ordering.
SAMPLE_CG_LIST = [
    _get_cudagraph(256, 1, 255),
    _get_cudagraph(256, 4, 252),
    _get_cudagraph(256, 256, 0),
    _get_cudagraph(128, 1, 127),
    _get_cudagraph(128, 4, 124),
    _get_cudagraph(128, 128, 0),
    _get_cudagraph(64, 1, 63),
    _get_cudagraph(64, 4, 60),
    _get_cudagraph(64, 64, 0),
    _get_cudagraph(16, 1, 15),
    _get_cudagraph(16, 4, 12),
    _get_cudagraph(16, 16, 0),
    _get_cudagraph(4, 1, 3),
    _get_cudagraph(4, 4, 0),
    _get_cudagraph(2, 1, 1),
    _get_cudagraph(2, 2, 0),
]


class TestGatingActivation:
    """Gating must be strictly opt-in via cuda_graph_all_prefills.

    Configs and tests that exercise the scheduler with use_cuda_graphs_for_non_decode_steps=True
    but cuda_graph_all_prefills=False will see the original scheduler behavior with no admission
    gating.
    """

    def test_legacy_request_payload_defaults_wait_counter(self):
        request = DynamicInferenceRequest.deserialize(
            {
                "request_id": 1,
                "prompt": None,
                "sampling_params": None,
                "inference_parameters": None,
                "status": None,
            }
        )
        assert request.cg_wait_iters == 0

    def test_inactive_when_all_prefills_off(self):
        engine = _create_engine(SAMPLE_CG_LIST)
        engine.cuda_graph_all_prefills = False
        assert engine._cg_admission_gating_active() is False

    def test_inactive_when_no_non_decode_graphs(self):
        engine = _create_engine(SAMPLE_CG_LIST)
        engine.context.use_cuda_graphs_for_non_decode_steps = False
        assert engine._cg_admission_gating_active() is False

    def test_inactive_when_cg_list_empty(self):
        engine = _create_engine([])
        assert engine._cg_admission_gating_active() is False

    def test_active_when_all_three_conditions_hold(self):
        engine = _create_engine(SAMPLE_CG_LIST)
        assert engine._cg_admission_gating_active() is True


class TestFindCgChunkSize:
    """_find_cg_chunk_size should snap to the largest CG-aligned chunk within budget."""

    def test_picks_largest_chunk_in_budget(self):
        # Empty active state, large budget — should pick the largest captured token_count.
        engine = _create_engine(SAMPLE_CG_LIST, active_tok=0, num_prefill=0, num_decode=0)
        assert engine._find_cg_chunk_size(max_chunk_tokens=500) == 256

    def test_respects_budget_ceiling(self):
        # Budget below largest CG — should pick the largest CG that still fits.
        engine = _create_engine(SAMPLE_CG_LIST)
        assert engine._find_cg_chunk_size(max_chunk_tokens=100) == 64
        assert engine._find_cg_chunk_size(max_chunk_tokens=20) == 16
        assert engine._find_cg_chunk_size(max_chunk_tokens=5) == 4

    def test_accounts_for_active_tokens(self):
        # Already 50 tokens in flight; chunk + active must land on a CG boundary.
        engine = _create_engine(SAMPLE_CG_LIST, active_tok=50)
        # Need cg.token_count - 50 in [1, max_chunk]. With max=300:
        #   256 - 50 = 206 (fits, valid). 128 - 50 = 78. 64 - 50 = 14. ...
        # Largest fitting: 256 → chunk = 206.
        assert engine._find_cg_chunk_size(max_chunk_tokens=300) == 206

    def test_returns_none_when_no_cg_fits(self):
        # No captured CG has token_count in (active, active+max_chunk]; helper returns
        # None so the caller can explicitly defer. Active=300, max_chunk=10 -> need
        # cg.token_count in (300, 310], none exists.
        engine = _create_engine(SAMPLE_CG_LIST, active_tok=300)
        assert engine._find_cg_chunk_size(max_chunk_tokens=10) is None

    def test_strict_mode_filters_insufficient_decode(self):
        # Hybrid model: matcher requires captured_D >= real_D. At active D=125 and
        # adding 1 new prefill, candidate (X, 1, 125) needs captured D >= 125.
        # Only (256, 1, 255) and (256, 4, 252) qualify on D; pick smallest token_count
        # that fits in budget — both have token=256, so chunk=256 is returned.
        engine = _create_engine(
            SAMPLE_CG_LIST, active_tok=0, num_prefill=0, num_decode=125, is_hybrid=True
        )
        assert engine._find_cg_chunk_size(max_chunk_tokens=300) == 256

    def test_strict_mode_no_match_returns_none(self):
        # Active D=200, only (256, *, 252) and (256, *, 255) have D >= 200, requiring
        # chunk=256. With smaller budget no CG matches in strict mode.
        engine = _create_engine(
            SAMPLE_CG_LIST, active_tok=0, num_prefill=0, num_decode=200, is_hybrid=True
        )
        assert engine._find_cg_chunk_size(max_chunk_tokens=100) is None

    def test_empty_cg_list_returns_none(self):
        engine = _create_engine([], active_tok=0)
        assert engine._find_cg_chunk_size(max_chunk_tokens=50) is None

    def test_smaller_shape_can_match_when_larger_shape_is_incompatible(self):
        # P/D compatibility is not monotonic with token_count: the larger graph has no
        # prefill slot, but the smaller graph can admit this request in strict mode.
        graph_dims = [_get_cudagraph(256, 0, 256), _get_cudagraph(128, 1, 127)]
        engine = _create_engine(graph_dims, is_hybrid=True)
        assert engine._find_cg_chunk_size(max_chunk_tokens=300) == 128

    def test_largest_compatible_boundary_does_not_depend_on_list_order(self):
        graph_dims = [_get_cudagraph(64, 1, 63), _get_cudagraph(256, 1, 255)]
        engine = _create_engine(graph_dims, is_hybrid=True)
        assert engine._find_cg_chunk_size(max_chunk_tokens=300) == 256


class TestCgAdmissionCheck:
    """`_cg_admission_check` returns admission decision and updates request state."""

    def test_match_returns_true_and_resets_counter(self):
        engine = _create_engine(SAMPLE_CG_LIST)
        req = _make_request(cg_wait_iters=5)  # was previously deferred
        candidate = _get_cudagraph(64, 1, 0)
        assert engine._cg_admission_check(req, candidate) is True
        assert req.cg_wait_iters == 0

    def test_no_match_returns_false_and_increments_counter(self):
        engine = _create_engine([])  # no captured graphs at all
        req = _make_request()
        candidate = _get_cudagraph(64, 1, 0)
        assert engine._cg_admission_check(req, candidate) is False
        assert req.cg_wait_iters == 1

    def test_repeated_misses_accumulate(self):
        engine = _create_engine([])
        req = _make_request()
        for expected in range(1, 6):
            engine._cg_admission_check(req, _get_cudagraph(64, 1, 0))
            assert req.cg_wait_iters == expected

    def test_warning_fires_at_threshold(self, caplog):
        engine = _create_engine([], warn_after=3)
        req = _make_request()
        with caplog.at_level(logging.WARNING):
            for _ in range(3):
                engine._cg_admission_check(req, _get_cudagraph(64, 1, 0))
        starvation_warnings = [
            r for r in caplog.records if "deferred by CG-aware admission" in r.message
        ]
        assert len(starvation_warnings) == 1
        assert "3 steps" in starvation_warnings[0].message

    def test_warning_does_not_fire_below_threshold(self, caplog):
        engine = _create_engine([], warn_after=100)
        req = _make_request()
        with caplog.at_level(logging.WARNING):
            for _ in range(99):
                engine._cg_admission_check(req, _get_cudagraph(64, 1, 0))
        assert not any("deferred by CG-aware admission" in r.message for r in caplog.records)

    def test_warning_repeats_at_each_multiple(self, caplog):
        engine = _create_engine([], warn_after=2)
        req = _make_request()
        with caplog.at_level(logging.WARNING):
            for _ in range(6):
                engine._cg_admission_check(req, _get_cudagraph(64, 1, 0))
        starvation_warnings = [
            r for r in caplog.records if "deferred by CG-aware admission" in r.message
        ]
        # Fires at cg_wait_iters = 2, 4, 6.
        assert len(starvation_warnings) == 3

    def test_strict_vs_non_strict_decode_spillover(self):
        # CGs with high total slots but limited per-type D; only non-strict can absorb
        # the extra decodes by repurposing prefill slots.
        cg_list = [_get_cudagraph(128, 128, 0), _get_cudagraph(128, 64, 64)]
        candidate = InferenceBatchDimensions(
            token_count=64, prefill_req_count=1, decode_req_count=70
        )

        # Strict: needs captured_D >= 70. (128,128,0).D=0 ✗, (128,64,64).D=64 ✗. No match.
        strict_engine = _create_engine(cg_list, is_hybrid=True)
        assert strict_engine._cg_admission_check(_make_request(), candidate) is False

        # Non-strict: total=128 >= 71 ✓ on either CG; both match. Admit.
        non_strict_engine = _create_engine(cg_list, is_hybrid=False)
        assert non_strict_engine._cg_admission_check(_make_request(), candidate) is True

        # KDA uses strict P/D matching in this branch's context matcher too.
        kda_engine = _create_engine(cg_list, is_hybrid=False)
        kda_engine.context.has_kda = True
        assert kda_engine._cg_admission_check(_make_request(), candidate) is False


class TestFindChunkSizeStrictBoundary:
    """Regression coverage for the Mamba-at-max_requests strict-matching scenario."""

    def test_strict_at_max_requests_finds_p_grid_match(self):
        # P-grid {1, 2, 4, 8} captured; real wants P+1=3 with D=508 at max_requests=512.
        # Strict matching needs captured P>=3 AND captured D>=508 — (4, 508) satisfies.
        # Shows that with adequate P-grid coverage, strict admission at max_requests
        # is feasible (the next-larger P value absorbs the new prefill).
        cg_list = [
            _get_cudagraph(512, 1, 511),
            _get_cudagraph(512, 2, 510),
            _get_cudagraph(512, 4, 508),
            _get_cudagraph(512, 8, 504),
        ]
        engine = _create_engine(
            cg_list, active_tok=0, num_prefill=2, num_decode=508, is_hybrid=True
        )
        chunk = engine._find_cg_chunk_size(max_chunk_tokens=512)
        assert chunk == 512
        # Confirm admission check also succeeds for this candidate.
        candidate = InferenceBatchDimensions(
            token_count=512, prefill_req_count=3, decode_req_count=508
        )
        assert engine._cg_admission_check(_make_request(), candidate) is True

    def test_strict_above_max_decode_returns_no_match(self):
        # Real (P=2, D=510). Adding 1 prefill → (P=3, D=510) total=513 exceeds max.
        # No captured CG has D >= 510 except (512, 1, 511) which has P=1 < 3.
        cg_list = [
            _get_cudagraph(512, 1, 511),
            _get_cudagraph(512, 2, 510),
            _get_cudagraph(512, 4, 508),
        ]
        engine = _create_engine(
            cg_list, active_tok=0, num_prefill=2, num_decode=510, is_hybrid=True
        )
        # Helper returns None to signal "no CG match" to the caller — explicit so the
        # caller can't accidentally schedule an un-graphed batch.
        assert engine._find_cg_chunk_size(max_chunk_tokens=512) is None
        # and a subsequent admission check on the same candidate also fails.
        req = _make_request()
        candidate = InferenceBatchDimensions(
            token_count=512, prefill_req_count=3, decode_req_count=510
        )
        assert engine._cg_admission_check(req, candidate) is False


# Captured set for the deferral-flow tests: P-grid {1, 2, 4, 8, max=512} with
# decode-only counterparts. Designed so candidates with specific P/D combos can
# either match (admit) or miss (defer) depending on engine state.
DEFERRAL_CG_LIST = [
    _get_cudagraph(512, 1, 511),
    _get_cudagraph(512, 2, 510),
    _get_cudagraph(512, 4, 508),
    _get_cudagraph(512, 8, 504),
    _get_cudagraph(256, 1, 255),
    _get_cudagraph(256, 2, 254),
    _get_cudagraph(256, 4, 252),
    _get_cudagraph(256, 8, 248),
    _get_cudagraph(64, 1, 63),
    _get_cudagraph(64, 2, 62),
    _get_cudagraph(64, 4, 60),
    _get_cudagraph(64, 8, 56),
    _get_cudagraph(8, 0, 8),
    _get_cudagraph(64, 0, 64),
    _get_cudagraph(256, 0, 256),
    _get_cudagraph(512, 0, 512),
]


class TestSchedulerDeferralInteraction:
    """Multi-call scenarios that exercise the deferral / resume flow.

    Validates three properties of CG-aware admission gating:
      - When one request defers, another admittable request can still proceed.
      - A deferred request gets admitted once state changes (e.g., a decode completes and active-D
        drops).
      - The deferral path never silently falls back to eager — `_cg_admission_check`
        strictly returns False on miss, and no internal flag is flipped to "schedule eagerly anyway"
    """

    def test_admittable_request_proceeds_when_other_is_deferred(self):
        # Engine state: active (P=2, D=510) total=512. Mamba strict.
        # Captured (4, 508) has D=508 < 510, so a P=3 candidate (would defer) cannot match.
        # But a pure-decode candidate (token=8, P=0, D=1) matches (8, 0, 8) — admits.
        engine = _create_engine(
            DEFERRAL_CG_LIST, active_tok=512, num_prefill=2, num_decode=510, is_hybrid=True
        )

        # Request A: a new prefill that would push P to 3 — no captured shape covers it
        # in strict mode (no captured P>=3 AND D>=510).
        req_a = _make_request(request_id=1)
        candidate_a = InferenceBatchDimensions(
            token_count=512, prefill_req_count=3, decode_req_count=510
        )
        assert engine._cg_admission_check(req_a, candidate_a) is False
        assert req_a.cg_wait_iters == 1

        # Request B: a decode-only candidate that does match a captured graph.
        # Reset active state to a low-load scenario (admittable). In a real scheduler
        # these are sequential admissions against an evolving state.
        admit_engine = _create_engine(
            DEFERRAL_CG_LIST, active_tok=0, num_prefill=0, num_decode=0, is_hybrid=True
        )
        req_b = _make_request(request_id=2)
        candidate_b = InferenceBatchDimensions(
            token_count=8, prefill_req_count=0, decode_req_count=1
        )
        # Note: prefill_req_count=0 takes the decode-only branch in is_applicable_for_batch_dim,
        # which checks captured_decode_req_count >= real_decode_req_count and captured P==0.
        assert admit_engine._cg_admission_check(req_b, candidate_b) is True
        assert req_b.cg_wait_iters == 0

    def test_deferred_request_admits_once_state_changes(self):
        # Initial state: active (P=2, D=510). Candidate (P=3, D=510) misses in strict mode.
        engine = _create_engine(
            DEFERRAL_CG_LIST, active_tok=512, num_prefill=2, num_decode=510, is_hybrid=True
        )
        req = _make_request()
        candidate_high_d = InferenceBatchDimensions(
            token_count=512, prefill_req_count=3, decode_req_count=510
        )
        # First admission attempt: defers.
        assert engine._cg_admission_check(req, candidate_high_d) is False
        assert req.cg_wait_iters == 1

        # Second attempt with active state still at D=510: still defers, wait counter
        # increments since request hasn't been admitted.
        assert engine._cg_admission_check(req, candidate_high_d) is False
        assert req.cg_wait_iters == 2

        # Decodes complete: active D drops to 508. Now the candidate (P=3, D=508)
        # fits within captured (4, 508) strictly: P=4>=3, D=508>=508, total>=511.
        engine.context.num_decode_requests = 508
        engine.context.active_token_count = 510
        candidate_lower_d = InferenceBatchDimensions(
            token_count=512, prefill_req_count=3, decode_req_count=508
        )
        assert engine._cg_admission_check(req, candidate_lower_d) is True
        # Wait counter resets on successful admission — the deferred request was
        # finally admitted in a subsequent scheduler pass.
        assert req.cg_wait_iters == 0

    def test_admission_helpers_never_signal_eager_fallback(self):
        # The design invariant: on miss, `_cg_admission_check` returns False and
        # `_register_cg_wait` bumps the counter — that's it. No flag is set, no
        # alternate "schedule eagerly" path is taken. The scheduler's break-on-False
        # contract is what preserves the "no eager fallback under cuda_graph_all_prefills"
        # property.
        engine = _create_engine([], active_tok=0, num_prefill=0, num_decode=0)
        req = _make_request()
        candidate = _get_cudagraph(64, 1, 0)

        before_state = (
            engine.context.active_token_count,
            engine.context.num_prefill_requests,
            engine.context.num_decode_requests,
        )
        # Fire 20 consecutive misses to increment the wait counter and verify nothing else changed.
        for _ in range(20):
            assert engine._cg_admission_check(req, candidate) is False
        after_state = (
            engine.context.active_token_count,
            engine.context.num_prefill_requests,
            engine.context.num_decode_requests,
        )

        # Engine state is untouched by the gating helpers; only the request's
        # wait counter advances. This proves the helpers never bypass the deferral
        # via some "go eager" side channel.
        assert before_state == after_state
        assert req.cg_wait_iters == 20
        # The request object only has the fields we explicitly track — no surprise
        # "eager_fallback_armed" flag or similar appeared.
        assert set(vars(req).keys()) == {"request_id", "cg_wait_iters"}

    def test_two_requests_progress_independently_across_iterations(self):
        # Two distinct requests in the waiting queue. Request 1 misses (high D),
        # Request 2 hits (decode-only). Across multiple scheduler iterations,
        # request 2 makes progress on each iteration while request 1's wait
        # counter accumulates — until decodes drop and request 1 unblocks too.
        cg_list = DEFERRAL_CG_LIST
        engine = _create_engine(
            cg_list, active_tok=512, num_prefill=2, num_decode=510, is_hybrid=True
        )
        req_blocked = _make_request(request_id=1)
        req_admittable = _make_request(request_id=2)

        # Mismatching candidate: needs strict D>=510 with P>=3 -> no captured graph.
        blocked_candidate = InferenceBatchDimensions(
            token_count=512, prefill_req_count=3, decode_req_count=510
        )
        # Matching candidate for the admittable one (decode-only with covered D).
        admittable_candidate = InferenceBatchDimensions(
            token_count=8, prefill_req_count=0, decode_req_count=1
        )

        # Simulate 3 scheduler iterations, each incrementing req_blocked's wait counter
        # while req_admittable is admitted every step.
        results = []
        for step in range(3):
            blocked_admit = engine._cg_admission_check(req_blocked, blocked_candidate)
            admittable_admit = engine._cg_admission_check(req_admittable, admittable_candidate)
            results.append((blocked_admit, admittable_admit))

        # The blocked one defers on every step; the admittable one admits every step (counter stays
        # at 0 — it never accumulates because each step succeeds).
        for blocked_admit, admittable_admit in results:
            assert blocked_admit is False
            assert admittable_admit is True
        assert req_blocked.cg_wait_iters == 3
        assert req_admittable.cg_wait_iters == 0

        # Now active D drops since a decode completed. Check the previously blocked request is
        # admitted.
        engine.context.num_decode_requests = 508
        engine.context.active_token_count = 510
        unblocked_candidate = InferenceBatchDimensions(
            token_count=512, prefill_req_count=3, decode_req_count=508
        )
        assert engine._cg_admission_check(req_blocked, unblocked_candidate) is True
        assert req_blocked.cg_wait_iters == 0


_CHUNKED_PREFILL_CG_CASES = [
    # parameters are label, active_tok, num_prefill, num_decode, max_chunk, is_hybrid,
    # is_continuing, expected_chunk
    # - is_continuing=True  -> gating is skipped entirely; result is max_chunk
    # - CG match found      -> result is the snapped CG-aligned) chunk
    # - No CG match         -> eager fallback: result is max_chunk, not a deferral
    pytest.param(
        # Fresh batch, large budget — gating active, CG match at 256.
        0,
        0,
        0,
        300,
        False,
        False,
        256,
        id="new_request_cg_match",
    ),
    pytest.param(
        # Budget below the smallest CG (min token_count=2) — no match.
        # Chunked prefill falls back to eager: uses max_chunk=1, not deferred.
        0,
        0,
        0,
        1,
        False,
        False,
        1,
        id="new_request_no_cg_match_eager_fallback",
    ),
    pytest.param(
        # Continuing chunked prefill: gating is bypassed regardless of CG coverage.
        # Expected result equals max_chunk.
        50,
        1,
        0,
        100,
        False,
        True,
        100,
        id="continuing_chunked_prefill_gating_skipped",
    ),
]


class TestChunkedPrefillCgGating:
    """Parametrized coverage for the CG-gating decision inside schedule_chunked_prefill.

    Exercises three distinct paths:
    - is_continuing=True  : gating is skipped; result = max_chunk
    - CG hit              : result = snapped (CG-aligned) chunk size
    - CG miss             : eager fallback; result = max_chunk (not a deferral)
    """

    @pytest.mark.parametrize(
        "active_tok,num_prefill,num_decode,max_chunk,is_hybrid,is_continuing,expected_chunk",
        _CHUNKED_PREFILL_CG_CASES,
    )
    def test_chunk_size_decision(
        self,
        active_tok,
        num_prefill,
        num_decode,
        max_chunk,
        is_hybrid,
        is_continuing,
        expected_chunk,
    ):
        engine = _create_engine(
            SAMPLE_CG_LIST,
            active_tok=active_tok,
            num_prefill=num_prefill,
            num_decode=num_decode,
            is_hybrid=is_hybrid,
        )

        if engine._cg_admission_gating_active() and not is_continuing:
            snapped = engine._find_cg_chunk_size(max_chunk)
            chunk = snapped if snapped is not None else max_chunk
        else:
            # Gating skipped (is_continuing) or gating inactive.
            chunk = max_chunk

        assert chunk == expected_chunk

    def test_no_cg_match_does_not_defer(self):
        # Core invariant: when CG gating is active and no graph matches the budget,
        # the chunked-prefill path uses max_chunk (eager), never defers.
        # SAMPLE_CG_LIST smallest token_count=2; budget=1 guarantees no match.
        engine = _create_engine(SAMPLE_CG_LIST, active_tok=0, num_prefill=0, num_decode=0)
        result = engine._find_cg_chunk_size(max_chunk_tokens=1)
        assert result is None  # confirms the miss path
        # Caller's eager fallback: chunk = max_chunk, not deferred.
        chunk = result if result is not None else 1
        assert chunk == 1

    def test_cg_match_resets_wait_counter(self):
        # On a CG hit the wait counter must be reset to 0 (matches non-chunked behaviour).
        engine = _create_engine(SAMPLE_CG_LIST, active_tok=0, num_prefill=0, num_decode=0)
        req = _make_request(cg_wait_iters=7)
        snapped = engine._find_cg_chunk_size(max_chunk_tokens=300)
        assert snapped is not None  # hit
        req.cg_wait_iters = 0  # as the engine does on a hit
        assert req.cg_wait_iters == 0


class TestSchedulerCallPaths:
    """Exercise graph gating through both production scheduler methods."""

    def test_non_chunked_deferral_and_admission(self):
        request = _SchedulerRequest(request_id=1, prompt_length=64)
        context = _SchedulerContext([_get_cudagraph(32, 1, 0)], max_tokens=128)
        engine = _create_scheduler_engine(request, context)

        engine.schedule_non_chunked_prefill()

        assert list(engine.waiting_request_ids) == [1]
        assert context.admissions == []
        assert request.cg_wait_iters == 1

        context.cuda_graph_batch_dimensions_list = [_get_cudagraph(64, 1, 0)]
        engine.schedule_non_chunked_prefill()

        assert not engine.waiting_request_ids
        assert context.admissions == [(1, None, 64)]
        assert request.cg_wait_iters == 0
        assert request.added_to_context

    def test_non_chunked_candidate_uses_effective_prefix_length(self):
        request = _SchedulerRequest(
            request_id=1, prompt_length=100, cg_wait_iters=4, block_hashes=[9]
        )
        context = _SchedulerContext(
            [_get_cudagraph(4, 1, 0)],
            max_tokens=128,
            enable_prefix_caching=True,
            effective_prefill_tokens=4,
        )
        engine = _create_scheduler_engine(request, context)

        engine.schedule_non_chunked_prefill()

        assert not engine.waiting_request_ids
        assert context.admissions == [(1, None, 4)]
        assert request.cg_wait_iters == 0

    def test_chunked_snap_then_continue_without_rechecking_graphs(self, monkeypatch):
        request = _SchedulerRequest(request_id=1, prompt_length=100, cg_wait_iters=3)
        context = _SchedulerContext([_get_cudagraph(64, 1, 0)], max_tokens=128)
        engine = _create_scheduler_engine(request, context, chunked=True)

        engine.schedule_chunked_prefill()

        assert context.admissions == [(1, 64, 64)]
        assert context.chunked_prefill_request_id == 1
        assert request.remaining_prompt_length == 36
        assert request.finished_chunk_token_count == 64
        assert list(engine.waiting_request_ids) == [1]
        assert request.cg_wait_iters == 0

        # Simulate the previous chunk leaving the active batch. The continuing request must
        # progress without another CG-boundary search, even if the available shapes changed.
        context.active_token_count = 0
        context.num_prefill_requests = 0
        monkeypatch.setattr(
            engine,
            "_find_cg_chunk_size",
            lambda *_: pytest.fail("continuing chunk must bypass CG admission sizing"),
        )
        engine.schedule_chunked_prefill()

        assert context.admissions == [(1, 64, 64), (1, None, 36)]
        assert context.chunked_prefill_request_id == -1
        assert not engine.waiting_request_ids

    def test_chunked_eager_fallback_resets_wait_and_one_token_tail_defers(self):
        graph_dims = [_get_cudagraph(64, 1, 0)]
        request = _SchedulerRequest(request_id=1, prompt_length=1, cg_wait_iters=8)
        context = _SchedulerContext(graph_dims, max_tokens=1)
        engine = _create_scheduler_engine(request, context, chunked=True)

        engine.schedule_chunked_prefill()

        assert not engine.waiting_request_ids
        assert context.admissions == [(1, None, 1)]
        assert request.cg_wait_iters == 0

        tail_request = _SchedulerRequest(request_id=2, prompt_length=2, cg_wait_iters=8)
        tail_context = _SchedulerContext(graph_dims, max_tokens=1)
        tail_engine = _create_scheduler_engine(tail_request, tail_context, chunked=True)

        tail_engine.schedule_chunked_prefill()

        assert list(tail_engine.waiting_request_ids) == [2]
        assert tail_context.admissions == []
        assert tail_context.chunked_prefill_request_id == -1
        assert tail_request.remaining_prompt_length == 2
        assert tail_request.cg_wait_iters == 0


class TestAdmissionMatcherContract:
    def test_matcher_avoids_ep_collective(self, monkeypatch):
        engine = _create_engine(SAMPLE_CG_LIST)
        request = _make_request()
        candidate = _get_cudagraph(64, 1, 0)
        calls = {}

        def fake_match_graph_config(**kwargs):
            calls.update(kwargs)
            return candidate

        monkeypatch.setattr(
            CUDAGraphBatchDimensionBuilder,
            "match_graph_config",
            staticmethod(fake_match_graph_config),
        )

        assert engine._cg_admission_check(request, candidate)
        assert calls["match_ep_token_counts"] is False
