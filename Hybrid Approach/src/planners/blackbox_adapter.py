# src/planners/blackbox_adapter.py
from __future__ import annotations
from typing import List, Optional
from flatland.envs.rail_env import RailEnv
from flatland_blackbox.solvers.cbs import CBSSolver
from flatland_blackbox.solvers.pp import PrioritizedPlanningSolver
from flatland_blackbox.utils import add_proxy_nodes
from src.planners.token_utils import apply_tokens_to_graph

from src.planners.state_extraction import build_rail_digraph

def _get_priority_order(env: RailEnv, tokens: Optional[List[dict]] = None):
    """Return agents reordered so PRIORITY-token agents are planned first."""
    agents = list(env.agents)

    if not tokens:
        return agents

    priority_agents = []
    for t in tokens:
        if t.get("kind") == "PRIORITY" and t.get("agent") is not None:
            priority_agents.append(int(t["agent"]))

    seen = set()
    priority_agents = [a for a in priority_agents if not (a in seen or seen.add(a))]

    # Mapping handle -> agent
    handle_to_agent = {a.handle: a for a in agents}

    ordered = [handle_to_agent[h] for h in priority_agents if h in handle_to_agent]
    ordered += [a for a in agents if a.handle not in priority_agents]

    return ordered


def plan_cbs(env: RailEnv, tokens: Optional[List[dict]] = None):
    G = build_rail_digraph(env)
    print(f"[plan_cbs] graph before tokens: nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    apply_tokens_to_graph(G, tokens)
    print(f"[plan_cbs] graph after tokens: nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    G = add_proxy_nodes(G, env.agents, cost=0.0)
    print(f"[plan_cbs] graph after proxy nodes: nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    for a in env.agents:
        print(
            f"[plan_cbs] agent{a.handle}: "
            f"start={a.initial_position} target={a.target} "
            f"earliest_departure={a.earliest_departure}"
        )

    solver = CBSSolver(G)
    return solver.solve(env.agents)



def plan_pp(env: RailEnv, tokens: Optional[List[dict]] = None):
    ordered_agents = _get_priority_order(env, tokens)

    G = build_rail_digraph(env)
    apply_tokens_to_graph(G, tokens)
    G = add_proxy_nodes(G, ordered_agents, cost=0.0)

    print("[plan_pp] planning order:", [a.handle for a in ordered_agents])

    solver = PrioritizedPlanningSolver(G)
    return solver.solve(ordered_agents)