
########################### Works with all T3.4 JSON Scenarios ####################################
# This module provides a robust loader for T3.4-style JSON scenario files into Flatland's RailEnv format

import json
from typing import Optional, Tuple, Any, List, Dict

import numpy as np

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap
from flatland.envs.grid.rail_env_grid import RailEnvTransitions
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.timetable_utils import Line, Timetable
from flatland.envs.observations import GlobalObsForRailEnv

# Normally, Flatland would generate a random scenario.
# Since we want to use our own scenarios, we slightly trick Flatland
# using a helper and a wrapper.
#
# Conceptually:
# env.reset()
#    ↓
# Flatland calls rail_generator()
#    ↓
# We return our custom grid_map
#    ↓
# Flatland builds the environment using our grid
# -------------------------
# Robust parsing helpers
# -------------------------

def _unwrap_singletons(x: Any) -> Any:
    """Unwrap nested singleton lists: [[x]] -> x (recursively)."""
    while isinstance(x, list) and len(x) == 1:
        x = x[0]
    return x

def _is_pos_pair(x: Any) -> bool:
    return isinstance(x, list) and len(x) == 2 and all(isinstance(v, (int, float)) for v in x)

def _collect_positions(x: Any) -> List[Tuple[int, int]]:
    """
    The JSON files are extremely nested.
    This first part of the helper flattens structures like [[[1]]] into 1.
    Collect all (r,c) pairs found in arbitrarily nested list structures.

    Works for:
      [2,8]
      [[2,8]]
      [[[2,8]]]
      [[[[2,8]]]]
      [[[2,8]], [[3,8]]]
      [[[[2,8]]], [[[3,8]]]]
      and mixed structures.
    """
    x = _unwrap_singletons(x)

    if _is_pos_pair(x):
        return [(int(x[0]), int(x[1]))]

    if isinstance(x, list):
        out: List[Tuple[int, int]] = []
        for item in x:
            out.extend(_collect_positions(item))
        return out

    return []

def _collect_ints(x: Any) -> List[int]:
    """Collect ints found in arbitrarily nested list structures (order preserved)."""
    x = _unwrap_singletons(x)

    if isinstance(x, (int, float)):
        return [int(x)]

    if isinstance(x, list):
        out: List[int] = []
        for item in x:
            out.extend(_collect_ints(item))
        return out

    return []

def _as_list(x: Any) -> List[Any]:
    if isinstance(x, list):
        return x
    return [x]

def _align_list_length(values: List[Any], n: int, pad_with_last: bool = True) -> List[Any]:
    """Make list exactly length n (truncate or pad)."""
    if n <= 0:
        return []
    if len(values) == n:
        return values
    if len(values) > n:
        return values[:n]
    # pad
    if not values:
        return [None] * n
    pad_val = values[-1] if pad_with_last else None
    return values + [pad_val] * (n - len(values))


# --------------------------------
# Flatland generator wrapper helpers
# # Essentially, this says:
# "Here is my finished rail network. Please create a function
# that returns it later when requested."
# --------------------------------

def rail_generator_from_grid_map(grid_map: RailGridTransitionMap, level_free_positions: List[Tuple[int, int]]):
    def rail_generator(*args, **kwargs):
        return grid_map, {
            "agents_hints": {"city_positions": {}},
            "level_free_positions": level_free_positions,
        }
    return rail_generator

# Returns a fully constructed Line object.
def line_generator_from_line(line: Line):
    def line_generator(*args, **kwargs):
        return line
    return line_generator

# Returns a fully constructed Timetable.
def timetable_generator_from_timetable(timetable: Timetable):
    def timetable_generator(*args, **kwargs):
        return timetable
    return timetable_generator


# -------------------------
# Main robust loader
# -------------------------

def load_scenario_from_json_robust(
    scenario_path: str,
    observation_builder=None,
    max_agents: Optional[int] = None, # You can load fewer agents here if desired.
) -> RailEnv: # Type hint indicating that this function returns a Flatland RailEnv.
    """
    Loads your T3.4-style JSON variants into a Flatland RailEnv.

    Handles:
      - 0 agents (graph tests)
      - nested singleton lists
      - multi-stop agent_positions / agent_directions
      - targets with varying nesting
      - timetable arrays of different lengths
    """
    if observation_builder is None:
        observation_builder = GlobalObsForRailEnv()
        # Here, each agent receives global visibility of the grid
        # (the observation system is activated).
        # Without the observation builder, env.reset() would later crash.

    # The JSON file is opened, read, and converted into a Python dictionary.
    with open(scenario_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # --- Grid ---
    # The grid dimensions are extracted here.
    # They are required so Flatland knows the map size
    # (number of columns and rows).
    width = int(data["gridDimensions"]["cols"])
    height = int(data["gridDimensions"]["rows"])

    transitions = RailEnvTransitions()
    # Without this, Flatland would not understand how the rails are connected
    # or how agents are allowed to move (i.e., which directions are valid).
    grid_data = np.array(data["grid"], dtype=transitions.get_type())
    # Safety check to ensure that the grid matrix matches the grid dimensions.
    if grid_data.shape != (height, width):
        grid_data = grid_data.reshape((height, width))
    # Here, the Flatland rail network is created from the grid matrix,
    # the transitions, and the dimensions.
    # This essentially represents the map with all rail connections.
    grid_map = RailGridTransitionMap(
        width=width,
        height=height,
        transitions=transitions,
        grid=grid_data
    )

    # overpasses can be nested sometimes -> robust parse
    # In the JSON file, overpasses are stored as lists.
    # They are converted into tuples here.
    level_free_positions = [_collect_positions(item)[0] for item in data.get("overpasses", []) if _collect_positions(item)]

    # --- Agents raw ---
    # Raw representation before conversion to Flatland's internal format.
    fl_line = data.get("flatland line", {})
    raw_positions = fl_line.get("agent_positions", [])
    raw_targets = fl_line.get("agent_targets", [])
    raw_directions = fl_line.get("agent_directions", [])
    raw_speeds = fl_line.get("agent_speeds", [])

    n_available = len(raw_positions) 
    if n_available == 0:
        n_agents = 0 
    else:
        n_agents = n_available if max_agents is None else max_agents
        if not (1 <= n_agents <= n_available):
            raise ValueError(f"max_agents must be between 1 and {n_available}")

    # --- Build Line (waypoints + speeds) ---
    # start -> target (each agent has 2 stops)
    agent_waypoints: Dict[int, List[List[Waypoint]]] = {}
    agent_speeds: List[float] = []
    # Loop over all agents to extract their start position, target position,
    # direction, and speed, and convert them into the Flatland format.
    for i in range(n_agents):
        # positions: can represent start + intermediate stops (variable nesting)
        pos_seq = _collect_positions(raw_positions[i])
        if not pos_seq:
            raise ValueError(f"No positions found for agent {i} in {scenario_path}")

        # targets: often one final target, but nesting varies; take LAST found pos as target
        tgt_seq = _collect_positions(raw_targets[i]) if i < len(raw_targets) else []
        target = tgt_seq[-1] if tgt_seq else pos_seq[-1]  # fallback

        # ensure final stop equals target
        if pos_seq[-1] != target:
            pos_seq = pos_seq + [target]

        # directions: can be single, per-stop, or nested lists; align to pos_seq length
        dir_seq = _collect_ints(raw_directions[i]) if i < len(raw_directions) else []
        dir_seq = _align_list_length(dir_seq, len(pos_seq), pad_with_last=True)

        # build waypoint list: each stop is a list with exactly one Waypoint
        wp_list: List[List[Waypoint]] = []
        for p, d in zip(pos_seq, dir_seq):
            # if direction missing, default to 0
            d_val = int(d) if d is not None else 0
            wp_list.append([Waypoint(position=(int(p[0]), int(p[1])), direction=d_val)])

        agent_waypoints[i] = wp_list

        # speed
        sp = raw_speeds[i] if i < len(raw_speeds) else 1.0
        agent_speeds.append(float(sp))

    line = Line(agent_waypoints=agent_waypoints, agent_speeds=agent_speeds)

    # --- Timetable ---
    # IMPORTANT: Flatland expects list-of-list per agent:
    fl_tt = data.get("flatland timetable", {})
    raw_earliest = fl_tt.get("earliest_departures", [])
    raw_latest = fl_tt.get("latest_arrivals", [])
    max_episode_steps = int(fl_tt.get("max_episode_steps", 0))

    earliest_departures: List[List[int]] = []
    latest_arrivals: List[List[Optional[int]]] = []
    # For each agent, sanitize the departure and arrival times
    # so they match the Flatland format.
    for i in range(n_agents):
        # get per-agent timetable arrays (may be scalar, list, nested)
        ed_list = _as_list(raw_earliest[i]) if i < len(raw_earliest) else []
        la_list = _as_list(raw_latest[i]) if i < len(raw_latest) else []

        # We try to align timetable length with number of stops for that agent (waypoints)
        n_stops = len(agent_waypoints[i])

        # normalize values
        ed_vals = [(_unwrap_singletons(v)) for v in ed_list]
        la_vals = [(_unwrap_singletons(v)) for v in la_list]

        ed_vals = _align_list_length(ed_vals, n_stops, pad_with_last=True)
        la_vals = _align_list_length(la_vals, n_stops, pad_with_last=True)

        # Flatland expects ints for earliest_departures.
        # If None appears, set to 0 (meaning "can depart immediately") to avoid crashes.
        ed_out: List[int] = []
        for v in ed_vals:
            ed_out.append(int(v) if v is not None else 0)

        # latest_arrivals can be Optional[int]
        la_out: List[Optional[int]] = []
        for v in la_vals:
            la_out.append(int(v) if v is not None else None)

        earliest_departures.append(ed_out)
        latest_arrivals.append(la_out)
    # A proper Flatland Timetable object is created here.
    timetable = Timetable(
        earliest_departures=earliest_departures,
        latest_arrivals=latest_arrivals,
        max_episode_steps=max_episode_steps
    )
    # # The actual Flatland environment is created here using all the information
    # we extracted and converted from the JSON file.
    # This is essentially the final map, including all agents, their timetables,
    # rail connections, etc., which Flatland can then use.
    env = RailEnv(
        width=width,
        height=height,
        number_of_agents=n_agents,
        rail_generator=rail_generator_from_grid_map(grid_map, level_free_positions),
        line_generator=line_generator_from_line(line),
        timetable_generator=timetable_generator_from_timetable(timetable),
        obs_builder_object=observation_builder,
    )

    # optional stations (some jsons have it, some empty)
    if hasattr(env, "stations"):
        env.stations = data.get("stations", [])

    return env


def get_num_agents_robust(scenario_path: str) -> int:
    with open(scenario_path, "r", encoding="utf-8") as f:   # 'data' now contains the complete scenario information.
        data = json.load(f)
    fl_line = data.get("flatland line", {})
    return len(fl_line.get("agent_positions", []))          # Count the number of agents.

# Alias for older code that imports this function by the old name.
load_scenario_from_json = load_scenario_from_json_robust