from src.environments.scenario_loader import load_scenario_from_json
from flatland.envs.observations import TreeObsForRailEnv


if __name__ == "__main__":
    scenario_path = "src/environments/simple_avoidance.json"

    env = load_scenario_from_json(
        scenario_path,
        observation_builder=TreeObsForRailEnv(max_depth=2)
    )

    obs, info = env.reset()

    print("Scenario loaded")
    print("Agents:", env.get_num_agents())

    for step in range(20):
        actions = {agent: 0 for agent in range(env.get_num_agents())}

        obs, rewards, done, info = env.step(actions)

        print(f"Step {step} | Rewards: {rewards}")

        if done["__all__"]:
            print("Episode finished")
            break