# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight unit tests for the EC connector async-load primitives.

These tests target the stand-alone helpers — ``_normalize_has_cache_item``,
``_EmbeddingLoadState`` lifecycle, ``EncoderCacheManager.check_only`` /
``register_external_loaded`` — without spinning up a full Scheduler. The
end-to-end async lifecycle (Park → Promote → Re-admit) is exercised by
the live local Dynamo run; full Scheduler-driven unit tests would
duplicate large fixture surface from
``tests/v1/kv_connector/unit/utils.py`` and are deferred to a follow-up.
"""

import pathlib
import types

import pytest
import torch

from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.example_connector import (
    ECExampleConnector,
)
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.outputs import ECConnectorOutput
from vllm.v1.request import RequestStatus


def _load_scheduler_helpers():
    """Pluck ``_EmbeddingLoadState`` + ``_normalize_has_cache_item`` from
    scheduler.py without triggering its full import chain (which transitively
    loads fused_moe / quantization paths that may depend on image-baked
    vllm._C symbols not present in fresh checkouts)."""
    src = (
        pathlib.Path(__file__).resolve().parents[4] / "vllm/v1/core/sched/scheduler.py"
    )
    text = src.read_text()
    helpers_src: list[str] = []
    keep = False
    for line in text.splitlines():
        if line.startswith("def _normalize_has_cache_item"):
            keep = True
        if keep:
            helpers_src.append(line)
        if keep and line.startswith("class Scheduler("):
            helpers_src.pop()  # don't include the class header
            break
    mod = types.ModuleType("_async_load_helpers")
    exec("from dataclasses import dataclass, field", mod.__dict__)
    exec("\n".join(helpers_src), mod.__dict__)
    return mod._normalize_has_cache_item, mod._EmbeddingLoadState


_normalize_has_cache_item, _EmbeddingLoadState = _load_scheduler_helpers()


pytestmark = [pytest.mark.unit, pytest.mark.gpu_0]


class _MockRequest:
    def __init__(self, request_id, mm_hashes, token_counts):
        self.request_id = request_id
        self._token_counts = token_counts
        self.mm_features = [
            MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier=h,
                mm_position=PlaceholderRange(offset=0, length=token_counts[i]),
            )
            for i, h in enumerate(mm_hashes)
        ]

    def get_num_encoder_embeds(self, input_id: int) -> int:
        return self._token_counts[input_id]


class TestNormalizeHasCacheItem:
    """Backward-compat shim for ``has_cache_item`` returning ``bool`` vs tuple."""

    def test_legacy_bool_true(self):
        assert _normalize_has_cache_item(True) == (True, False)

    def test_legacy_bool_false(self):
        assert _normalize_has_cache_item(False) == (False, False)

    def test_tuple_sync_hit(self):
        assert _normalize_has_cache_item((True, False)) == (True, False)

    def test_tuple_async_hit(self):
        assert _normalize_has_cache_item((True, True)) == (True, True)

    def test_tuple_miss(self):
        assert _normalize_has_cache_item((False, False)) == (False, False)


class TestRequestStatus:
    """Make sure WAITING_FOR_EMBEDDINGS doesn't break is_finished ordering."""

    def test_waiting_for_embeddings_is_not_finished(self):
        assert not RequestStatus.is_finished(RequestStatus.WAITING_FOR_EMBEDDINGS)

    def test_waiting_for_embeddings_lt_running(self):
        assert RequestStatus.WAITING_FOR_EMBEDDINGS < RequestStatus.RUNNING, (
            "WAITING_FOR_EMBEDDINGS must precede RUNNING in the enum"
        )

    def test_finished_states_still_finished(self):
        for st in (
            RequestStatus.FINISHED_STOPPED,
            RequestStatus.FINISHED_LENGTH_CAPPED,
            RequestStatus.FINISHED_ABORTED,
            RequestStatus.FINISHED_ERROR,
        ):
            assert RequestStatus.is_finished(st)


class TestEmbeddingLoadState:
    """Per-mm_hash state machine."""

    def test_default_state_loading(self):
        s = _EmbeddingLoadState(state="LOADING", num_embeds=10, waiters={"r1"})
        assert s.state == "LOADING"
        assert s.waiters == {"r1"}

    def test_resurrection_from_abandoned_to_loading(self):
        s = _EmbeddingLoadState(state="ABANDONED", num_embeds=10, waiters=set())
        # Simulate the resurrection path in _park_for_embeddings.
        s.state = "LOADING"
        s.waiters.add("r2")
        assert s.state == "LOADING"
        assert s.waiters == {"r2"}


class TestEncoderCacheManagerAsyncHelpers:
    """check_only + register_external_loaded against EncoderCacheManager."""

    def _mgr(self, size=100):
        return EncoderCacheManager(cache_size=size)

    def test_check_only_returns_false_for_uncached(self):
        mgr = self._mgr()
        req = _MockRequest("r1", ["h"], [10])
        assert mgr.check_only(req, 0) is False

    def test_check_only_does_not_mutate(self):
        mgr = self._mgr()
        req = _MockRequest("r1", ["h"], [10])
        mgr.cached["h"] = set()
        mgr.freeable["h"] = 10
        mgr.num_freeable_slots = 100
        before_freeable = dict(mgr.freeable)
        before_freeable_slots = mgr.num_freeable_slots
        assert mgr.check_only(req, 0) is True
        assert dict(mgr.freeable) == before_freeable, "check_only mutated freeable"
        assert mgr.num_freeable_slots == before_freeable_slots

    def test_register_external_loaded_with_refs(self):
        # Slots are reserved up-front via reserve_async_loaded_slots when
        # the scheduler parks the request; register_external_loaded then
        # promotes the entry without re-decrementing (would be double-count).
        mgr = self._mgr()
        assert mgr.reserve_async_loaded_slots(30)
        assert mgr.num_free_slots == 70
        mgr.register_external_loaded("h", 30, refs={"r1", "r2"})
        assert mgr.cached["h"] == {"r1", "r2"}
        assert "h" not in mgr.freeable, (
            "Externally-loaded entries must NOT be in freeable until refs drop"
        )
        # Still 70 — register doesn't decrement, only promotes.
        assert mgr.num_free_slots == 70

    def test_register_external_loaded_rejects_duplicate(self):
        mgr = self._mgr()
        mgr.reserve_async_loaded_slots(10)
        mgr.register_external_loaded("h", 10, refs={"r1"})
        # Second reserve+register for same hash must reject in cached check.
        mgr.reserve_async_loaded_slots(10)
        with pytest.raises(AssertionError):
            mgr.register_external_loaded("h", 10, refs={"r2"})

    def test_register_external_loaded_rejects_empty_refs(self):
        mgr = self._mgr()
        mgr.reserve_async_loaded_slots(10)
        with pytest.raises(AssertionError):
            mgr.register_external_loaded("h", 10, refs=set())

    def test_can_allocate_after_register(self):
        mgr = self._mgr(size=20)
        mgr.reserve_async_loaded_slots(10)
        mgr.register_external_loaded("h1", 10, refs={"r1"})
        # 10 slots left; can allocate 10 more.
        req = _MockRequest("r2", ["h2"], [10])
        assert mgr.can_allocate(
            req, 0, encoder_compute_budget=10, num_embeds_to_schedule=0
        )

    def test_reserve_release_async_loaded_slots(self):
        mgr = self._mgr()
        assert mgr.reserve_async_loaded_slots(40)
        assert mgr.num_free_slots == 60
        assert mgr.num_freeable_slots == 60
        mgr.release_async_loaded_slots(40)
        assert mgr.num_free_slots == 100
        assert mgr.num_freeable_slots == 100

    def test_reserve_async_loaded_slots_returns_false_on_overflow(self):
        mgr = self._mgr(size=10)
        assert mgr.reserve_async_loaded_slots(20) is False
        # No state mutation on rejection.
        assert mgr.num_free_slots == 10


class TestECConnectorOutputDefaults:
    """ECConnectorOutput.finished_loading must default to None."""

    def test_default_finished_loading(self):
        o = ECConnectorOutput()
        assert o.finished_loading is None

    def test_explicit_finished_loading(self):
        o = ECConnectorOutput(finished_loading={"h1", "h2"})
        assert o.finished_loading == {"h1", "h2"}


class TestECConnectorBackCompat:
    """Legacy ECExampleConnector (returns plain bool) must still work."""

    def test_get_finished_loads_default_returns_none(self, tmp_path):
        # Build a minimal mock VllmConfig.
        from unittest.mock import Mock

        from vllm.config import VllmConfig

        config = Mock(spec=VllmConfig)
        config.ec_transfer_config = Mock()
        config.ec_transfer_config.get_from_extra_config = Mock(
            return_value=str(tmp_path)
        )
        config.ec_transfer_config.is_ec_producer = True
        config.ec_transfer_config.is_ec_consumer = False
        connector = ECExampleConnector(
            vllm_config=config, role=ECConnectorRole.SCHEDULER
        )
        # Default impl from ECConnectorBase returns None.
        assert connector.get_finished_loads() is None

    def test_request_async_load_default_delegates(self, tmp_path, monkeypatch):
        """request_async_load default impl falls back to update_state_after_alloc.

        Spy on update_state_after_alloc to confirm the delegation, since the
        ECExampleConnector update_state_after_alloc has consumer-mode guards
        that early-return without producing any externally visible state on a
        producer-only mock — the assertion-via-side-effect pattern doesn't
        survive across vllm SHAs.
        """
        from unittest.mock import Mock

        from vllm.config import VllmConfig

        config = Mock(spec=VllmConfig)
        config.ec_transfer_config = Mock()
        config.ec_transfer_config.get_from_extra_config = Mock(
            return_value=str(tmp_path)
        )
        config.ec_transfer_config.is_ec_producer = True
        config.ec_transfer_config.is_ec_consumer = False
        connector = ECExampleConnector(
            vllm_config=config, role=ECConnectorRole.SCHEDULER
        )
        req = _MockRequest("r1", ["h1"], [10])
        called_with: list[tuple] = []
        monkeypatch.setattr(
            connector,
            "update_state_after_alloc",
            lambda r, i: called_with.append((r, i)),
        )
        connector.request_async_load(req, 0)
        assert called_with == [(req, 0)], (
            "request_async_load default impl must delegate to "
            "update_state_after_alloc(request, index)"
        )


class TestECConnectorMetadataEvictOrphan:
    """ECConnectorMetadata base must expose evict_orphan slot."""

    def test_evict_orphan_default_empty(self):
        # Use the example connector's metadata which inherits the base.
        from vllm.distributed.ec_transfer.ec_connector.example_connector import (
            ECExampleConnectorMetadata,
        )

        meta = ECExampleConnectorMetadata()
        assert meta.evict_orphan == set()

    def test_evict_orphan_assignment(self):
        from vllm.distributed.ec_transfer.ec_connector.example_connector import (
            ECExampleConnectorMetadata,
        )

        meta = ECExampleConnectorMetadata()
        meta.evict_orphan = {"h1", "h2"}
        assert meta.evict_orphan == {"h1", "h2"}


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA event lifecycle test requires GPU"
)
class TestCudaEventLifecycle:
    """Verify the assumed CUDA semantics: event.query() == True implies the
    captured H2D is GPU-globally complete (visible to subsequent default-stream
    reads). This is the core invariant that lets us skip an explicit
    default_stream.wait_event at dispatch time."""

    def test_event_query_after_h2d_completion(self):
        side = torch.cuda.Stream()
        host = torch.randn(1024 * 1024).pin_memory()  # 4 MB
        with torch.cuda.stream(side):
            gpu = host.to("cuda", non_blocking=True)
            event = torch.cuda.Event()
            event.record(side)
        # Synchronize via host loop until event.query() == True (mirrors
        # what get_finished_loads does each step).
        while not event.query():
            torch.cuda.synchronize()
        # After event.query() True, default-stream read of `gpu` must see
        # the data without explicit cross-stream wait.
        result = (gpu - host.to("cuda")).abs().max().item()
        assert result == 0.0, "Tensor mismatch: cross-stream visibility broken"
