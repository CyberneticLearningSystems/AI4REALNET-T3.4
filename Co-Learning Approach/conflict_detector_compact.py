"""
Conflict Detector for Compact Junction Environment
==================================================
Detects potential future conflicts between agents based on:
- Current positions
- Current velocities (last moves)
- Projected paths
- Track topology
"""

import numpy as np
from typing import List, Tuple, Set


class ConflictDetector:
    """Detects potential conflicts between agents"""
    
    def __init__(self, env):
        self.env = env
        # Track last moves to estimate direction
        self.last_moves = {}  # agent_id -> last_action
        
    def update_last_move(self, agent_id: int, action: int):
        """Record agent's last action for velocity estimation"""
        self.last_moves[agent_id] = action
    
    def predict_next_positions(self, agent_id: int, steps_ahead: int = 3) -> List[Tuple[int, int]]:
        """
        Predict possible positions for agent in next N steps.
        Returns list of (row, col) positions.
        """
        if self.env.agents_done[agent_id]:
            return []
        
        current_pos = tuple(self.env.agents_pos[agent_id])
        goal = self.env.agent_configs[agent_id][2:4]
        
        # Simple prediction: assume agent moves toward goal
        positions = [current_pos]
        
        pos = list(current_pos)
        for _ in range(steps_ahead):
            # Calculate direction toward goal
            dr = goal[0] - pos[0]
            dc = goal[1] - pos[1]
            
            # Determine most likely next move
            if abs(dr) > abs(dc):
                # More row distance - prefer North/South
                action = 0 if dr < 0 else 2  # North or South
            elif abs(dc) > 0:
                # More col distance - prefer East/West
                action = 1 if dc > 0 else 3  # East or West
            else:
                # At goal
                break
            
            # Apply action
            new_pos = pos.copy()
            if action == 0:  # North
                new_pos[0] -= 1
            elif action == 1:  # East
                new_pos[1] += 1
            elif action == 2:  # South
                new_pos[0] += 1
            elif action == 3:  # West
                new_pos[1] -= 1
            
            # Check if valid
            if (0 <= new_pos[0] < self.env.rows and 
                0 <= new_pos[1] < self.env.cols and
                self.env.grid[new_pos[0], new_pos[1]] != 0):
                positions.append(tuple(new_pos))
                pos = new_pos
            else:
                # Can't continue in this direction
                break
        
        return positions
    
    def detect_conflicts(self, lookahead: int = 3) -> List[dict]:
        """
        Detect potential conflicts between agents.
        
        Returns list of conflict dicts:
        {
            'agents': [id1, id2],
            'type': 'head_on' | 'same_cell' | 'swap',
            'positions': [(r1,c1), (r2,c2)],
            'steps_ahead': int,
            'severity': 'high' | 'medium' | 'low'
        }
        """
        conflicts = []
        
        # Get active agents
        active_agents = [i for i in range(self.env.n_agents) 
                        if not self.env.agents_done[i]]
        
        if len(active_agents) < 2:
            return []
        
        # Predict paths for all active agents
        predicted_paths = {}
        for agent_id in active_agents:
            predicted_paths[agent_id] = self.predict_next_positions(agent_id, lookahead)
        
        # Check for conflicts between each pair
        for i, agent1 in enumerate(active_agents):
            for agent2 in active_agents[i+1:]:
                path1 = predicted_paths[agent1]
                path2 = predicted_paths[agent2]
                
                # Check each time step
                for step in range(1, min(len(path1), len(path2))):
                    pos1 = path1[step]
                    pos2 = path2[step]
                    
                    # Same cell conflict
                    if pos1 == pos2:
                        conflicts.append({
                            'agents': [agent1, agent2],
                            'type': 'same_cell',
                            'positions': [pos1, pos2],
                            'steps_ahead': step,
                            'severity': 'high' if step <= 2 else 'medium'
                        })
                        break  # Only report first conflict per pair
                    
                    # Swap conflict (agents exchange positions)
                    if step > 0 and pos1 == path2[step-1] and pos2 == path1[step-1]:
                        conflicts.append({
                            'agents': [agent1, agent2],
                            'type': 'swap',
                            'positions': [pos1, pos2],
                            'steps_ahead': step,
                            'severity': 'high'
                        })
                        break
        
        return conflicts
    
    def is_at_junction(self, agent_id: int) -> bool:
        """Check if agent is currently at a junction"""
        if self.env.agents_done[agent_id]:
            return False
        
        pos = self.env.agents_pos[agent_id]
        track_type = self.env.grid[pos[0], pos[1]]
        return track_type == 3  # Type 3 = junction
    
    def is_critical_decision_point(self) -> Tuple[bool, List[dict]]:
        """
        Determine if this is a critical decision point requiring human input.
        
        Critical = potential conflict detected AND at least one agent at junction
        
        Returns: (is_critical, conflicts)
        """
        # Detect conflicts
        conflicts = self.detect_conflicts(lookahead=4)
        
        if not conflicts:
            return False, []
        
        # Check if any involved agent is at a junction
        for conflict in conflicts:
            for agent_id in conflict['agents']:
                if self.is_at_junction(agent_id):
                    # Critical! Conflict + junction = decision point
                    return True, conflicts
        
        return False, conflicts
    
    def get_conflict_description(self, conflict: dict) -> str:
        """Generate human-readable conflict description"""
        agent1, agent2 = conflict['agents']
        pos = conflict['positions'][0]
        
        colors = ['Red', 'Cyan', 'Yellow']
        
        if conflict['type'] == 'same_cell':
            return (f"⚠️ {colors[agent1]} A{agent1} and {colors[agent2]} A{agent2} "
                   f"will collide at ({pos[0]},{pos[1]}) in {conflict['steps_ahead']} steps!")
        elif conflict['type'] == 'swap':
            return (f"⚠️ {colors[agent1]} A{agent1} and {colors[agent2]} A{agent2} "
                   f"will swap positions (head-on) in {conflict['steps_ahead']} steps!")
        else:
            return f"⚠️ Potential conflict between A{agent1} and A{agent2}"
    
    def get_all_warnings(self) -> List[str]:
        """Get all current conflict warnings (for display, not for stopping)"""
        conflicts = self.detect_conflicts(lookahead=5)
        return [self.get_conflict_description(c) for c in conflicts]


if __name__ == "__main__":
    # Test the conflict detector
    from compact_junction_env import CompactJunctionEnv
    
    env = CompactJunctionEnv()
    detector = ConflictDetector(env)
    
    print("Testing Conflict Detector")
    print("=" * 60)
    print()
    
    # Test 1: Initial state (no conflict expected initially)
    print("Test 1: Initial state")
    is_critical, conflicts = detector.is_critical_decision_point()
    print(f"Critical: {is_critical}")
    print(f"Conflicts: {len(conflicts)}")
    print()
    
    # Test 2: Move agents toward each other
    print("Test 2: Agents moving toward collision")
    env.agents_pos = [[1, 1], [1, 3], [2, 0]]  # A0 and A1 close on main route
    
    is_critical, conflicts = detector.is_critical_decision_point()
    print(f"Critical: {is_critical}")
    print(f"Conflicts detected: {len(conflicts)}")
    
    if conflicts:
        for conflict in conflicts:
            print(f"  - {detector.get_conflict_description(conflict)}")
    
    warnings = detector.get_all_warnings()
    print(f"\nWarnings: {len(warnings)}")
    for warning in warnings:
        print(f"  {warning}")
    
    print()
    print("✓ Conflict detector test complete!")