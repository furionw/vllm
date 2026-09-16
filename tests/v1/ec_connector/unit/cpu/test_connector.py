# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock

from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.distributed.ec_transfer.ec_connector.cpu.common import (
    ECCPUConnectorMetadata,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.connector import ECCPUConnector
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory


def _cfg():
    ec = MagicMock()
    ec.is_ec_producer = True
    ec.is_ec_consumer = True
    cfg = MagicMock()
    cfg.ec_transfer_config = ec
    return cfg


def test_scheduler_role_builds_only_scheduler(monkeypatch):
    fake_sched = MagicMock()
    monkeypatch.setattr(ECCPUConnector, "_make_scheduler", lambda self, cfg: fake_sched)
    c = ECCPUConnector(_cfg(), ECConnectorRole.SCHEDULER)
    assert c.connector_scheduler is fake_sched
    assert c.connector_worker is None
    c.has_cache_item("x")
    fake_sched.has_cache_item.assert_called_once_with("x")


def test_worker_role_builds_only_worker(monkeypatch):
    fake_worker = MagicMock()
    monkeypatch.setattr(ECCPUConnector, "_make_worker", lambda self, cfg: fake_worker)
    c = ECCPUConnector(_cfg(), ECConnectorRole.WORKER)
    assert c.connector_worker is fake_worker
    assert c.connector_scheduler is None

    metadata = ECCPUConnectorMetadata(loads={"h": [0]})
    c.bind_connector_metadata(metadata)
    cache: dict = {}
    c.finish_load_caches(cache)
    fake_worker.finish_load_caches.assert_called_once_with(cache)


def test_worker_reports_hashes_scheduled_for_external_load(monkeypatch):
    monkeypatch.setattr(ECCPUConnector, "_make_worker", lambda self, cfg: MagicMock())
    c = ECCPUConnector(_cfg(), ECConnectorRole.WORKER)
    metadata = ECCPUConnectorMetadata(loads={"a": [0], "b": [1]})

    assert tuple(c.externally_loaded_hashes()) == ()
    c.bind_connector_metadata(metadata)
    assert set(c.externally_loaded_hashes()) == {"a", "b"}


def test_request_finished_inherited_noop(monkeypatch):
    monkeypatch.setattr(
        ECCPUConnector, "_make_scheduler", lambda self, cfg: MagicMock()
    )
    c = ECCPUConnector(_cfg(), ECConnectorRole.SCHEDULER)
    assert c.request_finished(MagicMock()) == (False, None)


def test_factory_registered():
    cls = ECConnectorFactory._registry["ECCPUConnector"]()
    assert cls is ECCPUConnector
