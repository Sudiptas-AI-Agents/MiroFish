"""Progress log for single-platform runs, in the format the parallel script writes."""

from action_logger import PlatformActionLogger
from run_parallel_simulation import fetch_new_actions_from_db, get_agent_names_from_config


class SimulationProgress:
    def __init__(self, platform, simulation_dir, config, db_path):
        self._log = PlatformActionLogger(platform, simulation_dir)
        self._names = get_agent_names_from_config(config)
        self._db_path = db_path
        self._last_rowid = 0
        self._initial_posts = 0
        self.total_actions = 0
        self._log.log_simulation_start(config)
        self._log.log_round_start(0, 0)

    def initial_post(self, agent_id, content):
        self._log.log_action(
            round_num=0,
            agent_id=agent_id,
            agent_name=self._names.get(agent_id, f"Agent_{agent_id}"),
            action_type="CREATE_POST",
            action_args={"content": content},
        )
        self._initial_posts += 1
        self.total_actions += 1

    def initial_done(self):
        self._log.log_round_end(0, self._initial_posts)

    def round_start(self, round_num, simulated_hour):
        self._log.log_round_start(round_num, simulated_hour)

    def round_skipped(self, round_num):
        self._log.log_round_end(round_num, 0)

    def round_done(self, round_num):
        actions, self._last_rowid = fetch_new_actions_from_db(
            self._db_path, self._last_rowid, self._names
        )
        for action in actions:
            self._log.log_action(
                round_num=round_num,
                agent_id=action["agent_id"],
                agent_name=action["agent_name"],
                action_type=action["action_type"],
                action_args=action["action_args"],
            )
        self.total_actions += len(actions)
        self._log.log_round_end(round_num, len(actions))

    def finish(self, total_rounds):
        self._log.log_simulation_end(total_rounds, self.total_actions)
