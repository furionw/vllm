# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
ECConnectorBase Class for Distributed Encoder Cache &
P2P Encoder cache communication in V1

The class provides the following primitives:
    Scheduler-side: runs in the scheduler, binds metadata, which
    is used by the worker-side to load/save Encoder cache.
        check_caches_exist() - Check whether Encoder cache of requests exist
        update_state_after_alloc() - update ECConnector state after
        allocate. This will decide to load the cache or not
        request_finished() - called when a request is finished,
        free the cache with the requests

    Worker-side: runs in each worker, loads/saves Encoder Cache to/from
    the Connector based on the metadata.
        start_load_ec() - starts loading all ECs (maybe async)
        wait_for_save() - blocks until all saves are done

        get_finished() - called with ids of finished requests, returns
            ids of requests that have completed async sending/recving.
"""

import enum
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ECConnectorOutput

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


class ECConnectorRole(enum.Enum):
    # Connector running in the scheduler process
    SCHEDULER = 0

    # Connector running in the worker process
    WORKER = 1


class ECConnectorMetadata(ABC):  # noqa: B024
    """
    Abstract Metadata used to communicate between the
    Scheduler ECConnector and Worker ECConnector.

    The base class declares an ``evict_orphan`` slot that the scheduler
    populates with mm_hashes whose async-load was abandoned after the H2D
    completed: the worker connector should drop these from any private
    "completed but unclaimed" tensor map. Concrete subclasses inherit this
    field via ``__post_init__`` (for dataclasses) or by calling
    ``super().__init__()``.
    """

    evict_orphan: set[str]

    def __init__(self) -> None:
        self.evict_orphan = set()


class ECConnectorBase(ABC):
    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole):
        self._connector_metadata: ECConnectorMetadata | None = None
        self._vllm_config = vllm_config
        self._role = role
        if vllm_config.ec_transfer_config is not None:
            self._is_producer = vllm_config.ec_transfer_config.is_ec_producer
            self._is_consumer = vllm_config.ec_transfer_config.is_ec_consumer
        else:
            raise ValueError("ec_transfer_config must be set for ECConnectorBase")

    @property
    def role(self) -> ECConnectorRole:
        return self._role

    @property
    def is_producer(self) -> bool:
        return self._is_producer

    @property
    def is_consumer(self) -> bool:
        return self._is_consumer

    # ==============================
    # Worker-side methods
    # ==============================

    def bind_connector_metadata(self, connector_metadata: ECConnectorMetadata) -> None:
        """Set the connector metadata from the scheduler.

        This function should be called by the model runner every time
        before the model execution. The metadata will be used for runtime
        EC cache loading.

        Args:
            connector_metadata (dict): the connector metadata.
        """
        self._connector_metadata = connector_metadata

    def clear_connector_metadata(self) -> None:
        """Clear the connector metadata.

        This function should be called by the model runner every time
        after the model execution.
        """
        self._connector_metadata = None

    def _get_connector_metadata(self) -> ECConnectorMetadata:
        """Get the connector metadata.

        This function should only be called inside the connector.

        Returns:
            ConnectorMetadata: the connector metadata.
        """

        # Should only be called while set to valid metadata.
        assert self._connector_metadata is not None
        return self._connector_metadata

    def register_caches(
        self,
        ec_caches: dict[str, torch.Tensor],
    ):
        """
        Initialize with the EC caches.
        Args:
            ec_caches: dictionary of encoder cache
        """
        # TODO: Implement this later for P2P feature
        return

    @abstractmethod
    def start_load_caches(
        self, encoder_cache: dict[str, torch.Tensor], **kwargs
    ) -> None:
        """
        Start loading the cache from the connector into vLLM's encoder cache.

        This method loads the encoder cache based on metadata provided by the scheduler.
        It is called before `_gather_mm_embeddings` for the EC Connector. For EC,
        the `encoder_cache` and `mm_hash` are stored in `kwargs`.

        Args:
            encoder_cache (dict[str, torch.Tensor]): A dictionary mapping multimodal
                data hashes (`mm_hash`) to encoder cache tensors.
            kwargs (dict): Additional keyword arguments for the connector.
        """
        pass

    @abstractmethod
    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs
    ) -> None:
        """
        Save the encoder cache to the connector.

        This method saves the encoder cache from the worker's local storage
        to shared storage or another external connector.

        Args:
            encoder_cache (dict[str, torch.Tensor]): A dictionary mapping multimodal
                data hashes (`mm_hash`) to encoder cache tensors.
            mm_hash (str): The hash of the multimodal data whose cache is being saved.
            kwargs (dict): Additional keyword arguments for the connector.
        """
        pass

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens on the worker.
        The scheduler process (via the Executors) will use this output
        to track which workers are done.

        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        return None, None

    def get_finished_loads(
        self,
        encoder_cache: dict[str, torch.Tensor] | None = None,
    ) -> set[str] | None:
        """Per-mm_hash async-load completion signal.

        Returns the set of mm_hashes whose async H2D copy event has fired
        since the last call. The connector MUST move each completed tensor
        into ``encoder_cache[mm_hash]`` before adding the hash to the
        returned set, so that the next consumer step's
        ``_gather_mm_embeddings`` sees the GPU-resident tensor.

        The default implementation returns ``None`` — connectors that don't
        support async loads (only the legacy sync block-fetch) keep returning
        ``None`` and never park requests via ``has_cache_item`` returning
        ``(True, True)``.

        Args:
            encoder_cache: the worker-side per-mm_hash GPU tensor map. The
                connector inserts completed tensors into this dict before
                reporting completion. Passed by ``ECConnectorModelRunnerMixin``.
        """
        return None

    # ==============================
    # Scheduler-side methods
    # ==============================

    @abstractmethod
    def has_cache_item(
        self,
        identifier: str,
    ) -> "bool | tuple[bool, bool]":
        """Check if a single encoder cache exists.

        Args:
            identifier: the identifier of the media (mm_hash).

        Returns:
            Either a plain ``bool`` (legacy: True means CPU/external hit, the
            scheduler routes this to the synchronous external_load path) or a
            tuple ``(hit, load_async)``:

            - ``(False, _)``   — miss; encoder runs.
            - ``(True, False)``— sync hit; today's behavior (worker
              ``start_load_caches`` performs an inline H2D in the same step
              the request is admitted).
            - ``(True, True)`` — async hit; the scheduler parks the request
              in ``RequestStatus.WAITING_FOR_EMBEDDINGS`` and dispatches the
              worker H2D on a side stream. The connector MUST report
              completion via ``get_finished_loads`` once the H2D's CUDA event
              has fired.

            Connectors that do not support async loads should keep returning
            plain ``bool`` (or ``(hit, False)``) — back-compat is guaranteed.
        """
        pass

    @abstractmethod
    def update_state_after_alloc(self, request: "Request", index: int):
        """Sync external-load hook.

        Called by the scheduler after a sync external-load hit
        (``has_cache_item`` returned ``True`` or ``(True, False)``) and
        ``encoder_cache_manager.allocate(request, i)`` has been called.

        Contract:
        - The connector MUST deduplicate by mm_hash. The scheduler may call
          this hook multiple times within a single step for the same
          ``(request, mm_hash)`` pair (once per sibling request sharing the
          hash), and the resulting connector metadata's ``loads`` collection
          MUST yield each mm_hash at most once. This is the system-level
          "exactly one H2D per hash" guarantee.

        Args:
            request: the request object.
            index: the mm_features index.
        """
        pass

    def request_async_load(self, request: "Request", index: int) -> None:
        """Async external-load hook.

        Called by the scheduler after a ``(True, True)`` hit, when the
        request is being parked in ``WAITING_FOR_EMBEDDINGS``. Unlike
        ``update_state_after_alloc``, ``encoder_cache_manager.allocate``
        has NOT been called for the hash — the slot reservation is tracked
        separately by the scheduler's inflight-async budget until the load
        completes and is promoted via ``register_external_loaded``.

        Default impl delegates to ``update_state_after_alloc`` so connectors
        with a unified load-recording path (e.g. dump everything into
        ``_loads_this_step``) work without override. Connectors that need
        to distinguish sync-allocated vs async-pending should override.
        """
        self.update_state_after_alloc(request, index)

    @abstractmethod
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        pass

    def update_connector_output(self, connector_output: ECConnectorOutput):
        """
        Update ECConnector state from worker-side connectors output.

        Args:
            connector_output (ECConnectorOutput): the worker-side
                connectors output.
        """
        return

    def request_finished(
        self, request: "Request"
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Called when a request has finished, before its encoder cache is freed.

        Returns:
            True if the request is being saved/sent asynchronously and cached
            should not be freed until the request_id is returned from
            get_finished().
        """
        return False, None
