import json
import numpy as np
from typing import Optional, List, Any

from flatland.envs.rail_env import RailEnv
from flatland.envs.timetable_utils import Line, Timetable
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap
from flatland.envs.grid.rail_env_grid import RailEnvTransitions
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.observations import GlobalObsForRailEnv


def rail_generator_from_grid_map(grid_map, level_free_positions):
    def rail_generator(*args, **kwargs):
        return grid_map, {
            "agents_hints": {"city_positions": {}},
            "level_free_positions": level_free_positions,
        }
    return rail_generator


def line_generator_from_line(line):
    def line_generator(*args, **kwargs):
        return line
    return line_generator


def timetable_generator_from_timetable(timetable):
    def timetable_generator(*args, **kwargs):
        return timetable
    return timetable_generator


# ------------------------
# ROBUST HELPERS
# ------------------------

def extract_positions(obj: Any) -> List[List[int]]:
    """Extract all [row, col] pairs from arbitrarily nested lists."""
    positions = []

    if isinstance(obj, list):
        if len(obj) == 2 and all(isinstance(x, (int, float)) for x in obj):
            positions.append([int(obj[0]), int(obj[1])])
        else:
            for item in obj:
                positions.extend(extract_positions(item))

    return positions


def normalize_target(obj: Any) -> List[int]:
    """Take last coordinate found in nested structure."""
    positions = extract_positions(obj)
    if not positions:
        raise ValueError(f"Could not extract target from {obj}")
    return positions[-1]


# ------------------------
# MAIN LOADER
# ------------------------

def load_scenario_from_json_robust(
    scenario_path: str,
    observation_builder=None,
    max_agents: Optional[int] = None
) -> RailEnv:

    with open(scenario_path, "r") as f:
        data = json.load(f)

    width = data["gridDimensions"]["cols"]
    height = data["gridDimensions"]["rows"]

    available_agents = data["flatland line"]["agent_positions"]

    if max_agents is not None:
        available_agents = available_agents[:max_agents]

    # -------- Positions --------
    agent_positions = []
    for agent_path in available_agents:
        coords = extract_positions(agent_path)
        if not coords:
            raise ValueError(f"No valid positions found for agent path: {agent_path}")
        agent_positions.append(coords)

    # -------- Targets --------
    raw_targets = data["flatland line"]["agent_targets"]
    if max_agents is not None:
        raw_targets = raw_targets[:max_agents]

    agent_targets = [normalize_target(t) for t in raw_targets]

    # -------- Directions --------
    agent_directions = data["flatland line"]["agent_directions"]
    if max_agents is not None:
        agent_directions = agent_directions[:max_agents]

    # align directions length with positions
    for i in range(len(agent_positions)):
        if len(agent_directions[i]) < len(agent_positions[i]):
            last_dir = agent_directions[i][-1]
            while len(agent_directions[i]) < len(agent_positions[i]):
                agent_directions[i].append(last_dir)

    # -------- Speeds --------
    agent_speeds = data["flatland line"]["agent_speeds"]
    if max_agents is not None:
        agent_speeds = agent_speeds[:max_agents]

    # -------- Timetable --------
    earliest_departures = data["flatland timetable"]["earliest_departures"]
    latest_arrivals = data["flatland timetable"]["latest_arrivals"]

    if max_agents is not None:
        earliest_departures = earliest_departures[:max_agents]
        latest_arrivals = latest_arrivals[:max_agents]

    # fix length mismatch if needed
    while len(earliest_departures) < len(agent_positions):
        earliest_departures.append(earliest_departures[-1])
    while len(latest_arrivals) < len(agent_positions):
        latest_arrivals.append(latest_arrivals[-1])

    number_of_agents = len(agent_positions)

    # -------- Build line --------
    line = Line(
        agent_waypoints=get_waypoints(agent_positions, agent_targets, agent_directions),
        agent_speeds=agent_speeds,
    )

    timetable = Timetable(
        earliest_departures=earliest_departures,
        latest_arrivals=latest_arrivals,
        max_episode_steps=data["flatland timetable"]["max_episode_steps"],
    )

    # -------- Grid --------
    transitions = RailEnvTransitions()
    grid_data = np.array(data["grid"], dtype=transitions.get_type())

    grid = RailGridTransitionMap(
        width=width,
        height=height,
        transitions=transitions,
        grid=grid_data,
    )

    level_free_positions = [tuple(item) for item in data["overpasses"]]

    if observation_builder is None:
        observation_builder = GlobalObsForRailEnv()

    env = RailEnv(
        width=width,
        height=height,
        number_of_agents=number_of_agents,
        rail_generator=rail_generator_from_grid_map(grid, level_free_positions),
        line_generator=line_generator_from_line(line),
        timetable_generator=timetable_generator_from_timetable(timetable),
        obs_builder_object=observation_builder,
    )

    env.stations = data["stations"]

    return env


# alias for compatibility
load_scenario_from_json = load_scenario_from_json_robust


def get_waypoints(agent_positions, agent_targets, agent_directions):
    agent_waypoints = {i: [] for i in range(len(agent_positions))}

    for agent, wpts in enumerate(agent_positions):
        for idx, wpt in enumerate(wpts):
            agent_waypoints[agent].append(
                [Waypoint(position=tuple(wpt), direction=agent_directions[agent][idx][0])]
            )

    for agent, wpt in enumerate(agent_targets):
        agent_waypoints[agent].append(
            [Waypoint(position=tuple(wpt), direction=agent_directions[agent][-1][0])]
        )

    return agent_waypoints


def get_num_agents(scenario_path: str) -> int:
    with open(scenario_path, "r") as f:
        data = json.load(f)
    return len(data["flatland line"]["agent_positions"])