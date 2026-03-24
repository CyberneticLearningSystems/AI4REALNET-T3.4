from flatland.envs.rail_env import RailEnv


class FlatlandEnvReference:
    def __init__(self, env: RailEnv = None):
        self.env = env

    def get_agent_handles(self):
        if self.env is not None:
            return self.env.get_agent_handles()
        return []
