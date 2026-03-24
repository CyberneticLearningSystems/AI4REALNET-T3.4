from pathlib import Path
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnvActions

from src.environments.scenario_loader import load_scenario_from_json

if __name__ == "__main__":
    scenario_path = Path("src/environments/simple_avoidance.json")                  
    # scenario_path = Path("src/environments/simple_avoidance_1.json")              
    # scenario_path = Path("src/environments/simple_avoidance_2.json")              
    # scenario_path = Path("src/environments/simple_ordering.json")                 
    # scenario_path = Path("src/environments/simple_ordering_for_vis.json")         
    # scenario_path = Path("src/environments/station_test.json")                    
    # scenario_path = Path("src/environments/overtaking.json")                      
    # scenario_path = Path("src/environments/network_experiments.json")             
    # scenario_path = Path("src/environments/graph_test_connected_schedules.json")  
    # scenario_path = Path("src/environments/graph_test_connected.json")            
    # scenario_path = Path("src/environments/graph_test.json")                          
    # scenario_path = Path("src/environments/complex_ordering.json")                
    # scenario_path = Path("src/environments/complex_avoidance.json")               
    # scenario_path = Path("src/environments/complex_avoidance_1.json")                        

    env = load_scenario_from_json(
        str(scenario_path),
        observation_builder=GlobalObsForRailEnv(),  
        max_agents=None
    )

    obs, info = env.reset()

    n = env.get_num_agents()
    print(f"Scenario loaded: {scenario_path.name}")
    print("Agents:", n, "| size:", env.width, "x", env.height)

    if n == 0:
        print("This scenario has 0 agents (graph/station test).")
    else:
        for i in range(min(n, 5)):  # change if you have scenarios with more than 5 agents and want to see them all
            print(f"agent{i} start:", env.agents[i].initial_position)
            print(f"agent{i} target:", env.agents[i].target)
            print(f"agent{i} earliest_departure:", env.agents[i].earliest_departure)

    for t in range(50):
        actions = {i: RailEnvActions.MOVE_FORWARD for i in range(env.get_num_agents())}

        obs, rewards, dones, infos = env.step(actions)

        print(f"\n=== t={t} ===")

        for i, agent in enumerate(env.agents):

            reached = (
                agent.position is not None and
                tuple(agent.position) == tuple(agent.target)
            )
            print(
                f"agent{i}: "
                f"pos={agent.position} "
                f"status={repr(agent.state)}"
                f"done={dones.get(i)} "
                f"reward={rewards.get(i)} "
                f"reached_target={reached} "
            )
        print("elapsed_steps:", env._elapsed_steps, "/", env._max_episode_steps)
        print("ALL DONE:", dones.get("__all__", False))

        if dones.get("__all__", False):
            break
        