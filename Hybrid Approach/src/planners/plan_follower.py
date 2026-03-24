# src/planners/plan_follower.py
from __future__ import annotations
from typing import Dict, Tuple, Any
from flatland.envs.rail_env import RailEnvActions

Node = Tuple[int, int, int]  # (r,c,dir)

def _action_to_reach_next(env, agent, next_rc):
    """Return a RailEnvActions that moves agent towards next_rc if possible, else STOP."""
    r, c = map(int, agent.position)
    cur_dir = int(agent.direction)

    # Get the integer encoding of the cell and derive the allowed exits for the current orientation from it
    cell = env.rail.get_full_transitions(r, c)
    trans = env.rail.transitions.get_transitions(cell, cur_dir)  # (N,E,S,W) 0/1

    candidates = []
    # forward
    fdir = cur_dir
    if trans[fdir] == 1:
        candidates.append((RailEnvActions.MOVE_FORWARD, fdir))
    # left
    ldir = (cur_dir - 1) % 4
    if trans[ldir] == 1:
        candidates.append((RailEnvActions.MOVE_LEFT, ldir))
    # right
    rdir = (cur_dir + 1) % 4
    if trans[rdir] == 1:
        candidates.append((RailEnvActions.MOVE_RIGHT, rdir))

    dir2delta = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    for act, nd in candidates:
        dr, dc = dir2delta[nd]
        if (r + dr, c + dc) == next_rc:
            return act

    return RailEnvActions.STOP_MOVING

def _dir_from_to(cur_rc, next_rc):
    (r, c) = cur_rc
    (nr, nc) = next_rc
    if nr == r - 1 and nc == c: return 0  # N
    if nr == r and nc == c + 1: return 1  # E
    if nr == r + 1 and nc == c: return 2  # S
    if nr == r and nc == c - 1: return 3  # W
    return None

def _action_from_dirs(cur_dir: int, next_dir: int):
    if next_dir is None:
        return RailEnvActions.STOP_MOVING
    if next_dir == cur_dir:
        return RailEnvActions.MOVE_FORWARD
    if next_dir == (cur_dir - 1) % 4:
        return RailEnvActions.MOVE_LEFT
    if next_dir == (cur_dir + 1) % 4:
        return RailEnvActions.MOVE_RIGHT
    # U-turn usually not allowed → stop
    return RailEnvActions.STOP_MOVING

def plan_to_actions(env, plan: Dict[int, Any], t: int) -> Dict[int, RailEnvActions]:
    """
    plan[agent_id] is assumed to be a list of (node, time) or similar.
    We'll try to read next node at time t+1; if not found => STOP.
    """
    actions: Dict[int, RailEnvActions] = {}
    n = env.get_num_agents()

    for i in range(n):
        agent = env.agents[i]

        # If no plan for this agent → STOP
        if plan is None or i not in plan:
            actions[i] = RailEnvActions.STOP_MOVING
            continue

        
        if agent.position is None:
            if env._elapsed_steps >= agent.earliest_departure:
                actions[i] = RailEnvActions.MOVE_FORWARD
            else:
                actions[i] = RailEnvActions.DO_NOTHING
            continue

        path = plan[i]

        # path could be list of tuples; try to locate entries by time
        # Normalize to dict time->node
        time_to_node = {}
        for item in path:
            if isinstance(item, tuple) and len(item) == 2:
                node, tt = item

                try:
                    tt = int(tt)
                except Exception:
                    continue

                # drop proxy nodes like (r,c,-1,agent_id)
                if isinstance(node, tuple) and len(node) == 4 and node[2] == -1:
                    continue

                time_to_node[tt] = node

        cur_pos = tuple(agent.position)
        cur_dir = int(agent.direction)



        plan_time = int(t) + 1

        times = sorted(time_to_node.keys())

        if len(times) < 2:
            actions[i] = RailEnvActions.STOP_MOVING
            continue

        # 2) Find the best-matching index:
        # preferred: a plan step whose (r, c) matches the current position and whose time <= plan_time
        cur_r, cur_c = cur_pos
        cur_idx = None

        for k in range(len(times) - 1, -1, -1):
            tt = times[k]
            node = time_to_node[tt]
            if isinstance(node, tuple) and len(node) == 3:
                r, c, _ = node
                if (r, c) == (cur_r, cur_c) and tt <= plan_time:
                    cur_idx = k
                    break

        # fallback: when not found take the first plan step >= plan_time
        if cur_idx is None:
            for k, tt in enumerate(times):
                if tt >= plan_time:
                    cur_idx = k - 1  # "current" ist then one of it
                    break
            if cur_idx is None:
                cur_idx = len(times) - 2  

        # 3) next node is cur_idx + 1
        if cur_idx + 1 >= len(times):
            actions[i] = RailEnvActions.STOP_MOVING
            continue

        next_node = time_to_node[times[cur_idx + 1]]
        if not (isinstance(next_node, tuple) and len(next_node) == 3):
            actions[i] = RailEnvActions.STOP_MOVING
            continue
        nr, nc, _ = next_node
        if (nr, nc) == (cur_r, cur_c):
            actions[i] = RailEnvActions.DO_NOTHING
            continue

        actions[i] = _action_to_reach_next(env, agent, (nr, nc))
        continue
    return actions