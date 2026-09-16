# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorBase
from vllm.v1.worker.ec_connector_model_runner_mixin import (
    ECConnectorModelRunnerMixin,
)

pytestmark = pytest.mark.cpu_test


def test_v1_finishes_load_before_model_execution_and_on_exit():
    connector = MagicMock(spec=ECConnectorBase)
    connector.is_consumer = True
    connector.get_finished.return_value = (None, None)
    scheduler_output = SimpleNamespace(
        ec_connector_metadata=object(), finished_req_ids=frozenset()
    )
    encoder_cache: dict = {}

    with (
        patch(
            "vllm.v1.worker.ec_connector_model_runner_mixin.get_ec_transfer",
            return_value=connector,
        ),
        ECConnectorModelRunnerMixin._get_ec_connector_output(
            scheduler_output, encoder_cache
        ),
    ):
        method_names = [call[0] for call in connector.mock_calls]
        assert method_names[-2:] == ["start_load_caches", "finish_load_caches"]

    method_names = [call[0] for call in connector.mock_calls]
    assert method_names[-3:] == [
        "finish_load_caches",
        "get_finished",
        "clear_connector_metadata",
    ]


def test_v1_cleanup_continues_when_final_load_drain_fails():
    connector = MagicMock(spec=ECConnectorBase)
    connector.is_consumer = True
    connector.get_finished.return_value = (None, None)
    connector.finish_load_caches.side_effect = [None, RuntimeError("drain failed")]
    scheduler_output = SimpleNamespace(
        ec_connector_metadata=object(), finished_req_ids=frozenset()
    )

    with (
        patch(
            "vllm.v1.worker.ec_connector_model_runner_mixin.get_ec_transfer",
            return_value=connector,
        ),
        pytest.raises(RuntimeError, match="drain failed"),
        ECConnectorModelRunnerMixin._get_ec_connector_output(scheduler_output, {}),
    ):
        pass

    connector.get_finished.assert_called_once_with(frozenset())
    connector.clear_connector_metadata.assert_called_once_with()


def test_v1_model_error_wins_over_cleanup_error():
    connector = MagicMock(spec=ECConnectorBase)
    connector.is_consumer = True
    connector.get_finished.return_value = (None, None)
    connector.finish_load_caches.side_effect = [None, RuntimeError("drain failed")]
    scheduler_output = SimpleNamespace(
        ec_connector_metadata=object(), finished_req_ids=frozenset()
    )

    with (
        patch(
            "vllm.v1.worker.ec_connector_model_runner_mixin.get_ec_transfer",
            return_value=connector,
        ),
        pytest.raises(ValueError, match="model failed"),
        ECConnectorModelRunnerMixin._get_ec_connector_output(scheduler_output, {}),
    ):
        raise ValueError("model failed")

    connector.get_finished.assert_called_once_with(frozenset())
    connector.clear_connector_metadata.assert_called_once_with()
