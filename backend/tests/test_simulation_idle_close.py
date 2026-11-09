import json
import os
import threading
import time
from types import SimpleNamespace

import pytest

from app.config import Config
from app.services import simulation_runner as runner_module
from app.services.simulation_ipc import SimulationIPCClient
from app.services.simulation_runner import (
    RunnerStatus,
    SimulationRunState,
    SimulationRunner,
)

SIM = "sim-idle"
LIMIT = 100


@pytest.fixture
def sim_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(Config, "SIMULATION_IDLE_CLOSE_SECONDS", LIMIT)
    path = tmp_path / SIM
    path.mkdir()
    _set_status(path, "alive")
    SimulationIPCClient(str(path))
    return path


def _set_status(path, status):
    (path / "env_status.json").write_text(json.dumps({"status": status}))


def _age(path, seconds):
    then = time.time() - seconds
    for item in (path / "env_status.json", path / "ipc_commands", path / "ipc_responses"):
        os.utime(item, (then, then))


def _respond_to_one_command(path, seen):
    """Stand in for the simulation script's side of the IPC folder."""

    def run():
        deadline = time.time() + 5
        while time.time() < deadline:
            for name in os.listdir(path / "ipc_commands"):
                command = json.loads((path / "ipc_commands" / name).read_text())
                seen.append(command["command_type"])
                (path / "ipc_responses" / name).write_text(
                    json.dumps({"command_id": command["command_id"], "status": "completed"})
                )
                return
            time.sleep(0.01)

    thread = threading.Thread(target=run)
    thread.start()
    return thread


def test_idle_seconds_is_none_while_rounds_are_running(sim_dir):
    _set_status(sim_dir, "running")
    _age(sim_dir, 500)

    assert SimulationIPCClient(str(sim_dir)).idle_seconds() is None


def test_idle_seconds_is_none_while_a_command_is_pending(sim_dir):
    (sim_dir / "ipc_commands" / "pending.json").write_text("{}")
    _age(sim_dir, 500)

    assert SimulationIPCClient(str(sim_dir)).idle_seconds() is None


def test_recent_command_activity_restarts_the_idle_clock(sim_dir):
    _age(sim_dir, 500)
    # Writing then removing a file is what an answered interview leaves behind.
    (sim_dir / "ipc_responses" / "done.json").write_text("{}")
    (sim_dir / "ipc_responses" / "done.json").unlink()

    assert SimulationIPCClient(str(sim_dir)).idle_seconds() < 10


def test_abandoned_environment_is_closed_through_ipc(sim_dir):
    _age(sim_dir, LIMIT + 5)
    seen = []
    responder = _respond_to_one_command(sim_dir, seen)

    SimulationRunner._close_if_idle(SIM, str(sim_dir))
    responder.join()

    assert seen == ["close_env"]


@pytest.mark.parametrize(
    "limit, age",
    [(LIMIT, LIMIT - 50), (0, 500)],
    ids=["not-idle-long-enough", "disabled"],
)
def test_environment_is_left_open(sim_dir, monkeypatch, limit, age):
    monkeypatch.setattr(Config, "SIMULATION_IDLE_CLOSE_SECONDS", limit)
    _age(sim_dir, age)

    SimulationRunner._close_if_idle(SIM, str(sim_dir))

    assert os.listdir(sim_dir / "ipc_commands") == []


def test_failed_close_does_not_escape_to_the_monitor(sim_dir, monkeypatch):
    _age(sim_dir, LIMIT + 5)

    def explode(_cls, _simulation_id):
        raise RuntimeError("close failed")

    monkeypatch.setattr(SimulationRunner, "close_simulation_env", classmethod(explode))

    SimulationRunner._close_if_idle(SIM, str(sim_dir))


def test_monitor_checks_for_idle_and_finishes_completed(sim_dir, monkeypatch):
    polls = iter([None, 0])
    process = SimpleNamespace(poll=lambda: next(polls), returncode=0)
    state = SimulationRunState(simulation_id=SIM, runner_status=RunnerStatus.RUNNING)
    checked = []
    monkeypatch.setattr(
        SimulationRunner, "get_run_state", classmethod(lambda _cls, _sim: state)
    )
    monkeypatch.setattr(SimulationRunner, "_save_run_state", classmethod(lambda _cls, _s: None))
    monkeypatch.setattr(
        SimulationRunner, "_sync_simulation_status", classmethod(lambda _cls, *_a, **_k: None)
    )
    monkeypatch.setattr(
        SimulationRunner,
        "_close_if_idle",
        classmethod(lambda _cls, sim, _path: checked.append(sim)),
    )
    monkeypatch.setattr(runner_module.time, "sleep", lambda _seconds: None)
    SimulationRunner._processes[SIM] = process

    SimulationRunner._monitor_simulation(SIM)

    assert checked == [SIM]
    assert state.runner_status == RunnerStatus.COMPLETED
