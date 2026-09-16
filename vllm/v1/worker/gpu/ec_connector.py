# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorBase
from vllm.logger import init_logger
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    ECConnectorOutput,
    ModelRunnerOutput,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache

logger = init_logger(__name__)


class ECConnector:
    """EC connector interface used by the V2 GPU model runner."""

    @contextmanager
    def maybe_get_output(
        self, scheduler_output: "SchedulerOutput"
    ) -> Generator[ECConnectorOutput | None, None, None]:
        yield None

    def no_forward(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput:
        return EMPTY_MODEL_RUNNER_OUTPUT

    def wait_for_loads(self) -> None:
        return None


class ActiveECConnector(ECConnector):
    def __init__(
        self,
        vllm_config: VllmConfig,
        encoder_cache: dict[str, torch.Tensor],
    ) -> None:
        self.encoder_cache = encoder_cache
        self.ec_connector = get_ec_transfer()
        assert isinstance(self.ec_connector, ECConnectorBase)
        # Every producer offloads freshly computed encoder outputs, including
        # an ec_both node that also reloads them.
        self.save_new_caches = self.ec_connector.is_producer
        self._loads_staged = False

    def wait_for_loads(self) -> None:
        if not self._loads_staged:
            return
        self.ec_connector.finish_load_caches(self.encoder_cache)
        self._loads_staged = False

    def _finish_step(
        self,
        output: ECConnectorOutput,
        scheduler_output: "SchedulerOutput",
    ) -> None:
        error: BaseException | None = None
        try:
            output.finished_sending, output.finished_recving = (
                self.ec_connector.get_finished(scheduler_output.finished_req_ids)
            )
        except BaseException as exc:
            error = exc

        try:
            output.ec_connector_worker_meta = (
                self.ec_connector.build_connector_worker_meta()
            )
        except BaseException as exc:
            if error is None:
                error = exc
            else:
                logger.exception("Failed to build EC connector worker metadata")

        try:
            self.ec_connector.clear_connector_metadata()
        except BaseException:
            if error is None:
                raise
            logger.exception("Failed to clear EC connector metadata")

        if error is not None:
            raise error.with_traceback(error.__traceback__)

    @contextmanager
    def maybe_get_output(
        self, scheduler_output: "SchedulerOutput"
    ) -> Generator[ECConnectorOutput | None, None, None]:
        if scheduler_output.ec_connector_metadata is None:
            yield None
            return

        output = ECConnectorOutput()
        ec_connector = self.ec_connector
        assert scheduler_output.ec_connector_metadata is not None
        ec_connector.bind_connector_metadata(scheduler_output.ec_connector_metadata)

        primary_error: BaseException | None = None
        try:
            try:
                if ec_connector.is_consumer:
                    self._loads_staged = True
                    ec_connector.start_load_caches(self.encoder_cache)

                cached_hashes = (
                    set(self.encoder_cache)
                    | set(ec_connector.externally_loaded_hashes())
                    if self.save_new_caches
                    else None
                )
                yield output
            except BaseException as exc:
                primary_error = exc
                try:
                    self.wait_for_loads()
                except BaseException:
                    logger.exception("Failed to finish EC loads after model failure")
                raise
            else:
                try:
                    self.wait_for_loads()
                    if cached_hashes is not None:
                        for mm_hash in self.encoder_cache.keys() - cached_hashes:
                            ec_connector.save_caches(
                                encoder_cache=self.encoder_cache, mm_hash=mm_hash
                            )
                except BaseException as exc:
                    primary_error = exc
                    raise
        finally:
            try:
                try:
                    self._finish_step(output, scheduler_output)
                except BaseException:
                    if primary_error is None:
                        raise
                    logger.exception("Failed to finalize EC connector step")
            finally:
                self._loads_staged = False

    def no_forward(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput:
        # EC send/recv even if no work to do.
        with self.maybe_get_output(scheduler_output) as ec_connector_output:
            pass

        return ModelRunnerOutput.with_ec_conn_output_only(ec_connector_output)


NO_OP_EC_CONNECTOR = ECConnector()


def get_ec_connector(
    vllm_config: VllmConfig,
    encoder_cache: "EncoderCache | None",
) -> ECConnector:
    if (
        not has_ec_transfer()
        or vllm_config.model_config.is_encoder_decoder
        or encoder_cache is None
    ):
        return NO_OP_EC_CONNECTOR

    return ActiveECConnector(vllm_config, encoder_cache.encoder_outputs)
