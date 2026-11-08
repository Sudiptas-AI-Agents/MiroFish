"""Single-platform runs must publish the progress SimulationRunner reads."""

import asyncio
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services.simulation_runner import SimulationRunner

SCRIPTS = Path(__file__).parents[1] / "scripts"
CASES = [
    ("reddit", "run_reddit_simulation", "RedditSimulationRunner", "generate_reddit_agent_graph",
     "reddit_profiles.json"),
    ("twitter", "run_twitter_simulation", "TwitterSimulationRunner", "generate_twitter_agent_graph",
     "twitter_profiles.csv"),
]


class FakeEnv:
    def __init__(self, graph):
        self.agent_graph = graph
        self.steps = []

    async def reset(self):
        pass

    async def step(self, actions):
        self.steps.append(actions)

    async def close(self):
        pass


class FakeIPC:
    def __init__(self, *args):
        pass

    def update_status(self, status):
        pass


@pytest.mark.parametrize("platform,module,runner_class,graph_factory,profile", CASES,
                         ids=[case[0] for case in CASES])
def test_run_writes_actions_and_simulation_end(
    monkeypatch, tmp_path, platform, module, runner_class, graph_factory, profile
):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    script = importlib.import_module(module)
    monkeypatch.setitem(sys.modules, module, script)

    config = {
        "simulation_id": "sim_test",
        "agent_configs": [{"agent_id": 0, "entity_name": "Alice"}],
        "time_config": {"total_simulation_hours": 2, "minutes_per_round": 60},
        "event_config": {"initial_posts": [{"poster_agent_id": 0, "content": "hello"}]},
    }
    config_path = tmp_path / "simulation_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / profile).write_text("[]", encoding="utf-8")

    agent = type("Agent", (), {})()
    graph = SimpleNamespace(get_agent=lambda agent_id: agent)
    env = FakeEnv(graph)

    async def fake_graph(**kwargs):
        return graph

    monkeypatch.setattr(script, graph_factory, fake_graph)
    monkeypatch.setattr(script.oasis, "make", lambda **kwargs: env)
    monkeypatch.setattr(script, "IPCHandler", FakeIPC)
    monkeypatch.setattr(getattr(script, runner_class), "_create_model", lambda self: None)
    # Round 1 has no active agents; round 2 has one.
    rounds = iter([[], [(0, agent)]])
    monkeypatch.setattr(
        getattr(script, runner_class),
        "_get_active_agents_for_round",
        lambda self, *args: next(rounds),
    )

    runner = getattr(script, runner_class)(str(config_path), wait_for_commands=False)
    asyncio.run(runner.run(max_rounds=2))

    log = tmp_path / platform / "actions.jsonl"
    events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    kinds = [event.get("event_type", "action") for event in events]
    assert kinds[0] == "simulation_start"
    assert kinds.count("round_start") == kinds.count("round_end") == 3
    assert [e["round"] for e in events if e.get("event_type") == "round_end"] == [0, 1, 2]
    initial = next(e for e in events if "event_type" not in e)
    assert initial["agent_name"] == "Alice" and initial["action_type"] == "CREATE_POST"
    assert events[-1]["event_type"] == "simulation_end"
    assert events[-1]["total_rounds"] == 2


def test_runner_treats_a_lone_platform_as_all_complete(monkeypatch, tmp_path):
    (tmp_path / "sim_x" / "reddit").mkdir(parents=True)
    (tmp_path / "sim_x" / "reddit" / "actions.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(tmp_path))
    state = SimpleNamespace(simulation_id="sim_x", twitter_completed=False, reddit_completed=True)

    assert SimulationRunner._check_all_platforms_completed(state) is True
