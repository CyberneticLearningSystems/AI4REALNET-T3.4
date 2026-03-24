"""

A simple environment with two possible bypasses design.

Grid: 3 rows × 5 columns
Agents: 3 agents with symmetrical conflicts
Routes: Main route + 2 parallel bypasses (upper & lower)

Agent Configuration:
- Agent 0: (1,0) → (1,4)  [Main route, left to right, 4 steps]
- Agent 1: (1,4) → (1,0)  [Main route, right to left, 4 steps] HEAD-ON!
- Agent 2: (2,0) → (0,4)  [Lower bypass to upper bypass, 6 steps] DIAGONAL!

Grid Topology:
  Col:  0    1    2    3    4
       
Row 0:     ┌─────────────┐── G2    Upper bypass
           │             │  
Row 1: A0──┼─────────────┼──A1   Main route
           │             │  
Row 2: A2──┴─────────────┘       Lower bypass

The topology is defined directly in this file, It can't load other Flatlands maps and Flatlands render function does not work

"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict


class CompactJunctionEnv(gym.Env):

    
    def __init__(self):
        super().__init__()
        
        # Grid dimensions
        self.rows = 3
        self.cols = 5
        self.n_agents = 3
        
        # Action space: 0=North, 1=East, 2=South, 3=West, 4=WAIT
        self.action_space = spaces.MultiDiscrete([5, 5, 5])
        
        # Observation space: [row, col, goal_dist, other1_dist, other2_dist, on_main, collision_risk]
        # 7 features × 3 agents = 21 features
        self.observation_space = spaces.Box(
            low=0, high=max(self.rows, self.cols),
            shape=(21,), dtype=np.float32
        )
        
        # Agent configurations: [start_r, start_c, goal_r, goal_c]
        self.agent_configs = [
            (1, 0, 1, 4),   # Agent 0: Left to right on main
            (1, 4, 1, 0),   # Agent 1: Right to left on main
            (2, 0, 0, 4),   # Agent 2: Lower-left to upper-right
        ]
        
        
        self.grid = self._build_grid()
        self.agents_pos = None
        self.agents_done = None
        self.step_count = 0
        self.max_steps = 30  
        
        # Initialize state
        self.reset()
        
    def _build_grid(self) -> np.ndarray:
        
        grid = np.zeros((self.rows, self.cols), dtype=np.uint8)
        
        # Track types: 0=empty, 1=horizontal, 2=vertical, 3=junction
        
        
        grid[0, 1] = 3  # Junction with main route
        grid[0, 2] = 1  # Horizontal
        grid[0, 3] = 3  # Junction with main route
        grid[0, 4] = 1  # Horizontal (goal for A2)
        
        # Row 1 - Main route
        grid[1, 0] = 1  # Start A0
        grid[1, 1] = 3  # Junction (main splits)
        grid[1, 2] = 1  # Horizontal
        grid[1, 3] = 3  # Junction (routes rejoin)
        grid[1, 4] = 1  # Start A1
        
        # Row 2 - Lower bypass
        grid[2, 0] = 1  # Start A2
        grid[2, 1] = 3  # Junction with main route
        grid[2, 2] = 1  # Horizontal
        grid[2, 3] = 3  # Junction with main route
        
        # Vertical connections
        # Column 1: Connects all 3 rows
        grid[0, 1] = 3
        grid[1, 1] = 3
        grid[2, 1] = 3
        
        # Column 3: Connects all 3 rows
        grid[0, 3] = 3
        grid[1, 3] = 3
        grid[2, 3] = 3
        
        return grid
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment"""
        # Gymnasium supports seed in reset()
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize agent positions
        self.agents_pos = []
        self.agents_done = []
        
        for config in self.agent_configs:
            start_r, start_c, _, _ = config
            self.agents_pos.append([start_r, start_c])
            self.agents_done.append(False)
        
        self.step_count = 0
        
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def _get_observation(self) -> np.ndarray:
        """Get observations"""
        obs = []
        
        for i in range(self.n_agents):
            if self.agents_done[i]:
                obs.extend([0] * 7)
                continue
            
            pos = self.agents_pos[i]
            goal = self.agent_configs[i][2:4]
            
            # Position
            obs.append(float(pos[0]))
            obs.append(float(pos[1]))
            
            # Distance to goal
            goal_dist = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
            obs.append(float(goal_dist))
            
            # Distance to other agents
            for j in range(self.n_agents):
                if j == i:
                    continue
                other_pos = self.agents_pos[j]
                dist = abs(pos[0] - other_pos[0]) + abs(pos[1] - other_pos[1])
                obs.append(float(dist))
            
            # On main route? (row 1 for A0/A1)
            on_main = 1.0 if pos[0] == 1 else 0.0
            obs.append(on_main)
            
            # Collision risk
            collision_risk = 0.0
            for j in range(self.n_agents):
                if j == i or self.agents_done[j]:
                    continue
                other_pos = self.agents_pos[j]
                dist = abs(pos[0] - other_pos[0]) + abs(pos[1] - other_pos[1])
                if dist <= 1:
                    collision_risk = 1.0
                    break
            obs.append(collision_risk)
        
        return np.array(obs, dtype=np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step"""
        self.step_count += 1
        total_reward = 0.0
        collision = False
        info = {}
        
        # Execute actions sequentially
        for agent_id in range(self.n_agents):
            if self.agents_done[agent_id]:
                continue
            
            agent_action = action[agent_id]
            reward, agent_collision = self._execute_action(agent_id, agent_action)
            total_reward += reward
            
            if agent_collision:
                collision = True
        
        # Check done
        done = all(self.agents_done) or self.step_count >= self.max_steps
        truncated = self.step_count >= self.max_steps and not all(self.agents_done)
        
        # Team bonuses - ONLY given when episode ends!
        if done:
            # Success bonus (all 3 reached goals)
            if all(self.agents_done) and not collision:
                total_reward += 100.0
                info['success'] = True
            else:
                info['success'] = False
            
            # Partial success bonus (2+ reached goals)
            num_done = sum(self.agents_done)
            if num_done >= 2 and not collision:
                total_reward += 20.0 * num_done
        else:
            info['success'] = False
        
        info['collision'] = collision
        info['num_done'] = sum(self.agents_done)
        
        obs = self._get_observation()
        
        return obs, total_reward, done, truncated, info
    
    def _execute_action(self, agent_id: int, action: int) -> Tuple[float, bool]:
        """Execute single agent action"""
        reward = -0.1
        collision = False
        
        if self.agents_done[agent_id]:
            return 0.0, False
        
        pos = self.agents_pos[agent_id]
        goal = self.agent_configs[agent_id][2:4]
        old_dist = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        
        # WAIT action
        if action == 4:
            reward -= 0.05
            return reward, False
        
        # Calculate new position
        new_pos = pos.copy()
        if action == 0:  # North
            new_pos[0] -= 1
        elif action == 1:  # East
            new_pos[1] += 1
        elif action == 2:  # South
            new_pos[0] += 1
        elif action == 3:  # West
            new_pos[1] -= 1
        
        # Check bounds
        if new_pos[0] < 0 or new_pos[0] >= self.rows or new_pos[1] < 0 or new_pos[1] >= self.cols:
            reward -= 1.0
            return reward, False
        
        # CRITICAL FIX: Check if current cell allows movement in this direction!
        # Type 0 = empty (invalid)
        # Type 1 = horizontal track (only East/West allowed)
        # Type 2 = vertical track (only North/South allowed)
        # Type 3 = junction (all directions allowed)
        current_track = self.grid[pos[0], pos[1]]
        
        # Validate direction based on current cell's track type
        if current_track == 1:  # Horizontal track
            if action in [0, 2]:  # North or South
                # Can't move vertically on horizontal track!
                reward -= 1.0
                return reward, False
        elif current_track == 2:  # Vertical track
            if action in [1, 3]:  # East or West
                # Can't move horizontally on vertical track!
                reward -= 1.0
                return reward, False
        # Type 3 (junction) allows all directions - no restriction
        
        # Check if target cell is valid (non-empty)
        target_track = self.grid[new_pos[0], new_pos[1]]
        if target_track == 0:
            reward -= 1.0
            return reward, False
        
        # Check collision
        for other_id in range(self.n_agents):
            if other_id == agent_id or self.agents_done[other_id]:
                continue
            other_pos = self.agents_pos[other_id]
            if new_pos[0] == other_pos[0] and new_pos[1] == other_pos[1]:
                reward -= 10.0
                collision = True
                return reward, collision
        
        # Move agent
        self.agents_pos[agent_id] = new_pos
        
        # Progress reward
        new_dist = abs(new_pos[0] - goal[0]) + abs(new_pos[1] - goal[1])
        if new_dist < old_dist:
            reward += 1.0
        elif new_dist > old_dist:
            reward -= 0.5
        
        # Check goal
        if new_pos[0] == goal[0] and new_pos[1] == goal[1]:
            reward += 50.0
            self.agents_done[agent_id] = True
        
        return reward, collision


if __name__ == "__main__":
    """Test the environment"""
    env = CompactJunctionEnv()
    obs, info = env.reset()
    
    print("=" * 70)
    print("COMPACT 3×5 JUNCTION ENVIRONMENT")
    print("=" * 70)
    print(f"Grid: {env.rows} rows × {env.cols} cols")
    print(f"Agents: {env.n_agents}")
    print()
    
    print("Agent Configurations:")
    for i, config in enumerate(env.agent_configs):
        start_r, start_c, goal_r, goal_c = config
        dist = abs(goal_r - start_r) + abs(goal_c - start_c)
        print(f"  Agent {i}: ({start_r},{start_c}) → ({goal_r},{goal_c}), Distance: {dist} steps")
    print()
    
    print("Grid Layout:")
    symbols = {0: "  ", 1: "──", 2: "║ ", 3: "╬ "}
    for r in range(env.rows):
        row_str = f"Row {r}: "
        for c in range(env.cols):
            # Check for agents
            agent_here = None
            for i, pos in enumerate(env.agents_pos):
                if pos[0] == r and pos[1] == c:
                    agent_here = i
                    break
            
            if agent_here is not None:
                row_str += f"A{agent_here} "
            else:
                row_str += symbols[env.grid[r, c]]
        print(row_str)
    
    print()
    print("=" * 70)