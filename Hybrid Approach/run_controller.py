# run_controller.py
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnvActions
from src.planners.blackbox_adapter import plan_cbs, plan_pp
from src.environments.scenario_loader import load_scenario_from_json

from src.planners.plan_follower import plan_to_actions

@dataclass
class Token:
    kind: str              # "PRIORITY", "AVOID_EDGE", "PREFER_LEFT", "SWITCH"
    agent: Optional[int]  
    payload: dict


class NegotiationManager:
    def __init__(self):
        self.tokens: List[Token] = []

    def add(self, token: Token):
        self.tokens.append(token)

    def consume_all(self) -> List[Token]:
        out = self.tokens[:]
        self.tokens.clear()
        return out

class HybridController:
    def __init__(self, mode: str = "cbs"):
        self.mode = mode
        self.current_plan = None
        self.last_positions = None
        self.no_move_counter = 0
        self.forced_wait = {}
        self.active_priority_agent = None
        self.release_schedule = {}
        

    def detect_deadlock(self, env) -> bool:
        positions = tuple(
            tuple(agent.position) if agent.position is not None else None
            for agent in env.agents
        )

        if self.last_positions == positions:
            self.no_move_counter += 1
        else:
            self.no_move_counter = 0

        self.last_positions = positions
        return self.no_move_counter >= 3

    def replan(self, env, tokens):
        token_dicts = [t.__dict__ for t in tokens] if tokens else []

        # Special case: all agents have the same start and the same target
        # -> break symmetry before planning
        same_start = env.get_num_agents() == 4 and all(
            a.initial_position == env.agents[0].initial_position for a in env.agents
        )
        same_target = env.get_num_agents() == 4 and all(
            a.target == env.agents[0].target for a in env.agents
        )

        if same_start and same_target:
            if not self.release_schedule:
                self.release_schedule = {0: 1, 1: 4, 2: 8, 3: 12}
                print(f"[replan] staggered departures for planning: {self.release_schedule}")

            for a in env.agents:
                if a.handle in self.release_schedule:
                    a.earliest_departure = self.release_schedule[a.handle]

        # Check if a PRIORITY Token exists
        has_priority = any(t.kind == "PRIORITY" for t in tokens)

        active_mode = "pp" if has_priority else "cbs"

    

        if has_priority:
            print("[replan] PRIORITY detected -> using PP")
            self.current_plan = plan_pp(env, token_dicts)
        else:
            self.current_plan = plan_cbs(env, token_dicts)
        
        if self.current_plan is None:
            print(f"[replan] mode={active_mode} -> no plan")
        else:
            print(f"[replan] mode={active_mode} plan keys={list(self.current_plan.keys())}")

    
            lens = {aid: len(path) for aid, path in self.current_plan.items()}
            print(f"[replan] plan lengths={lens}")
            print("[debug] agent0 plan first 12:", self.current_plan.get(0, [])[:12])
            print("[debug] agent1 plan first 12:", self.current_plan.get(1, [])[:12])

    def act(self, env, t: int, tokens: List[Token]) -> Dict[int, RailEnvActions]:
        if self.current_plan is None or tokens:
            self.replan(env, tokens)
                            
        actions = plan_to_actions(env, self.current_plan, t)

        # Special logic for complex_ordering:
        # all agents have the same start and the same target -> take turns
        same_start = env.get_num_agents() == 4 and all(
            a.initial_position == env.agents[0].initial_position for a in env.agents
        )
        same_target = env.get_num_agents() == 4 and all(
            a.target == env.agents[0].target for a in env.agents
        )

        if same_start and same_target:
            active_agent = None
            for i, a in enumerate(env.agents):
                if getattr(a, "state", None) != 6: 
                    active_agent = i
                    break

            if active_agent is not None:
                for i in range(env.get_num_agents()):
                    if i != active_agent:
                        actions[i] = RailEnvActions.STOP_MOVING

                print(f"[act] complex_ordering sequence -> only agent{active_agent} may move")


        priority_agents = [tok.agent for tok in tokens if tok.kind == "PRIORITY" and tok.agent is not None]

        if priority_agents:
            self.active_priority_agent = priority_agents[0]

        if self.active_priority_agent is not None:
            p = self.active_priority_agent
            p_agent = env.agents[p]

            p_pos = tuple(p_agent.position) if p_agent.position is not None else None
            p_done = getattr(p_agent, "state", None) == 6

            # If the priority agent is finished -> release the gate
            if p_done:
                self.active_priority_agent = None

            # As long as the priority agent has not progressed far enough -> others wait
            elif p_pos is not None and p_pos[1] < 7:
                for i in range(env.get_num_agents()):
                    if i != p:
                        actions[i] = RailEnvActions.STOP_MOVING
                print(f"[act] priority gate active -> agent{p} goes first")

            else:
                self.active_priority_agent = None
        return actions

if __name__ == "__main__":
    scenario_path = Path("src/environments/simple_avoidance.json")                
    # scenario_path = Path("src/environments/simple_avoidance_1.json")              
    # scenario_path = Path("src/environments/simple_avoidance_2.json")              
    # scenario_path = Path("src/environments/simple_ordering.json")                 
    # scenario_path = Path("src/environments/simple_ordering_for_vis.json")                           
    # scenario_path = Path("src/environments/overtaking.json")                      
    # scenario_path = Path("src/environments/network_experiments.json")             
    # scenario_path = Path("src/environments/graph_test_connected_schedules.json")  
    # scenario_path = Path("src/environments/complex_avoidance.json")               
    # scenario_path = Path("src/environments/complex_avoidance_1.json")             
    env = load_scenario_from_json(str(scenario_path), observation_builder=GlobalObsForRailEnv(), max_agents=None)
    obs, info = env.reset()

    print(f"Scenario loaded: {scenario_path.name}")
    print("Agents:", env.get_num_agents(), "| size:", env.width, "x", env.height)

    if env.get_num_agents() == 0:
        print("This scenario has 0 agents. No planning required.")
        raise SystemExit(0)

    for i, agent in enumerate(env.agents):
        print(f"agent{i} start:", agent.initial_position)
        print(f"agent{i} target:", agent.target)
        print(f"agent{i} earliest_departure:", agent.earliest_departure)

    negotiator = NegotiationManager()
    controller = HybridController(mode="cbs")  # Start with the CBS algorithm and switch to PP when a PRIORITY token is received

    for t in range(50):
        # Priority logic: "agent0 has higher priority"
        #if t == 2 and "ordering" in scenario_path.name: 
        if t == 2 and scenario_path.name in ["simple_ordering.json", "simple_ordering_for_vis.json"]:
            negotiator.add(Token(kind="PRIORITY", agent=0, payload={"boost": 10}))

        tokens = negotiator.consume_all()
        actions = controller.act(env, env._elapsed_steps, tokens)

        obs, rewards, dones, infos = env.step(actions)

        print(f"\n=== t={t} mode={controller.mode} tokens={len(tokens)} ===")
        for i, a in enumerate(env.agents):
            print(f"agent{i}: pos={a.position} state={a.state} reward={rewards.get(i)} done={dones.get(i)}")
        if dones.get("__all__", False):
            break

        # Works well for avoidance scenarios, but not for simple ordering, complex ordering, and simple_ordering_for_vis
        if t == 3 and "avoidance" in scenario_path.name:
            negotiator.add(
                Token(
                    kind="AVOID_EDGE",
                    agent=None,
                    payload={
                        "u": (2,4,1),
                        "v": (2,5,1)
                    }
                )
            )