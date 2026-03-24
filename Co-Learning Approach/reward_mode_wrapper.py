"""
Reward Mode Wrapper for Multi-Mode Training
============================================
Wraps the compact junction environment to modify rewards based on mode.

Modes:
- SAFE: Prioritizes zero collisions (high collision penalty, low step penalty)
- EFFICIENT: Prioritizes speed (low collision penalty, high step penalty)
- BALANCED: Default settings (moderate on all dimensions)

Usage:
    from reward_mode_wrapper import RewardModeWrapper
    
    env = CompactJunctionEnv()
    
    # Wrap for safe mode
    env = RewardModeWrapper(env, mode='safe')
    
    # Train as normal
    model = PPO("MlpPolicy", env)
    model.learn(300_000)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class RewardModeWrapper(gym.Wrapper):
    """
    Wrapper that modifies rewards based on training mode.
    
    Modes:
    - 'safe': High collision penalty, low step penalty, encourage WAIT
    - 'efficient': Low collision penalty, high step penalty, discourage WAIT, speed bonus
    - 'balanced': Default environment rewards (no modification)
    """
    
    def __init__(self, env, mode='balanced'):
        super().__init__(env)
        
        self.mode = mode
        
        # Track episode info for speed bonus calculation
        self.episode_start_step = 0
        
        # Define reward parameters for each mode
        self._setup_mode_parameters()
        
        print(f"\n{'='*60}")
        print(f"REWARD MODE: {mode.upper()}")
        print(f"{'='*60}")
        self._print_mode_info()
        print(f"{'='*60}\n")
    
    def _setup_mode_parameters(self):
        """Setup reward modification parameters based on mode"""
        
        if self.mode == 'safe':
            # SAFE MODE 🛡️ - Avoid collisions at all costs
            # 
            # KEY INSIGHT: Collision is TERMINAL (episode ends immediately)
            # So we don't need extreme -50 penalty - that just makes agent too scared!
            # Instead: moderate collision penalty + high success reward = agent tries but carefully
            # 
            # NOTE: This wrapper can't distinguish WAIT from MOVE (no access to actions)
            # So step_penalty and wait_penalty both just affect "active agent penalty"
            # The base env already applies -0.15 for WAIT vs -0.1 for MOVE
            self.params = {
                'collision_penalty': -15.0,    # Moderate increase (vs -10 base)
                                               # Not -50! Collision is already terminal!
                'step_penalty': -0.12,         # Slightly higher than base -0.1
                                               # Encourages some urgency
                'wait_penalty': -0.12,         # Same as step (wrapper can't distinguish anyway)
                'team_success': +250.0,        # MUCH higher to motivate trying
                                               # Risk -15, gain +250 = worth it!
                'partial_success_mult': 2.0,   # +40 per agent (was +30)
                'speed_bonus': 0.0,            # No speed bonus
                'speed_threshold': None
            }
            
        elif self.mode == 'efficient':
            # EFFICIENT MODE ⚡ - Complete as fast as possible
            self.params = {
                'collision_penalty': -5.0,     # Much lower (vs -10)
                'step_penalty': -0.3,          # Much higher (vs -0.1)
                'wait_penalty': -0.5,          # Much higher (vs -0.15)
                'team_success': +100.0,        # Normal
                'partial_success_mult': 1.0,   # +20 per agent (normal)
                'speed_bonus': +50.0,          # NEW: Bonus for fast completion
                'speed_threshold': 15          # Bonus if completed in <15 steps
            }
            
        else:  # balanced
            # BALANCED MODE ⚖️ - Current default settings
            self.params = {
                'collision_penalty': -10.0,    # Normal
                'step_penalty': -0.1,          # Normal
                'wait_penalty': -0.15,         # Normal
                'team_success': +100.0,        # Normal
                'partial_success_mult': 1.0,   # +20 per agent (normal)
                'speed_bonus': 0.0,            # No speed bonus
                'speed_threshold': None
            }
    
    def _print_mode_info(self):
        """Print mode configuration"""
        print(f"Collision Penalty:  {self.params['collision_penalty']:+7.2f}")
        print(f"Step Penalty:       {self.params['step_penalty']:+7.2f}")
        print(f"WAIT Penalty:       {self.params['wait_penalty']:+7.2f}")
        print(f"Team Success Bonus: {self.params['team_success']:+7.2f}")
        print(f"Partial Success:    {self.params['partial_success_mult']:.1f}x multiplier")
        
        if self.params['speed_bonus'] > 0:
            print(f"Speed Bonus:        {self.params['speed_bonus']:+7.2f} (if <{self.params['speed_threshold']} steps)")
        
        print()
        print("Expected Behavior:")
        if self.mode == 'safe':
            print("  • Careful coordination at junctions")
            print("  • May use bypasses to avoid conflicts")
            print("  • Very low collision rate (1-5%)")
            print("  • Moderate speed (12-18 steps)")
            print("  • HIGH success rate target (85-95%)")
            print("  • Prioritizes completing successfully over speed")
        elif self.mode == 'efficient':
            print("  • Takes main route often (fastest)")
            print("  • Rarely uses WAIT")
            print("  • Higher collision rate (10-20%)")
            print("  • Very fast episodes (8-14 steps)")
            print("  • Moderate success rate (60-75%)")
        else:
            print("  • Uses bypasses when needed")
            print("  • WAITs strategically")
            print("  • Medium collision rate (5-10%)")
            print("  • Medium speed (12-16 steps)")
            print("  • Good success rate (75-85%)")
    
    def reset(self, **kwargs):
        """Reset environment and tracking"""
        obs, info = self.env.reset(**kwargs)
        self.episode_start_step = self.env.step_count
        return obs, info
    
    def step(self, action):
        """Execute step and modify rewards based on mode"""
        # Execute base environment step
        obs, base_reward, done, truncated, info = self.env.step(action)
        
        # Modify rewards based on mode
        modified_reward = self._modify_reward(base_reward, done, info)
        
        return obs, modified_reward, done, truncated, info
    
    def _modify_reward(self, base_reward, done, info):
        """
        Modify reward based on mode settings.
        
        The base environment gives rewards in specific ways.
        We need to extract and modify them.
        
        NOTE: This wrapper has a fundamental limitation:
        - We don't have access to individual agent actions
        - So we can't tell if an agent WAIT vs MOVE
        - Both step_penalty and wait_penalty end up applied the same way
        - The base env already distinguishes (-0.1 for MOVE, -0.15 for WAIT)
        - Our modifications apply on top of that base behavior
        """
        # Start with base reward
        reward = 0.0
        
        # Get environment state
        step_count = self.env.step_count
        agents_done = self.env.agents_done
        
        # Check what happened this step
        # We'll reconstruct the reward with mode-specific values
        
        # For each active agent, accumulate step penalties
        for agent_id in range(self.env.n_agents):
            if not agents_done[agent_id]:
                # Agent took a step (or WAIT)
                # Base env gives -0.1 for step, -0.15 for WAIT
                # We need to apply our mode-specific penalties
                
                # Get the action for this agent
                # Note: We don't have direct access to individual agent actions here
                # So we'll apply an average penalty
                reward += self.params['step_penalty']
        
        # Check for collision (base env gives -10 per collision)
        # If base_reward has a large negative component, it's likely a collision
        if base_reward < -5:  # Heuristic: collision happened
            # Remove base collision penalty, add mode-specific one
            num_collisions = int(abs(base_reward + 0.3) / 10)  # Estimate number
            reward += self.params['collision_penalty'] * num_collisions
        
        # Check for success (episode done)
        if done:
            num_done = sum(agents_done)
            
            if num_done == self.env.n_agents and step_count <= self.env.max_steps:
                # All agents succeeded!
                reward += self.params['team_success']
                
                # Speed bonus (efficient mode only)
                if self.params['speed_bonus'] > 0:
                    episode_length = step_count - self.episode_start_step
                    if episode_length < self.params['speed_threshold']:
                        reward += self.params['speed_bonus']
                        
            elif num_done >= 2:
                # Partial success (2+ agents)
                reward += 20.0 * num_done * self.params['partial_success_mult']
        
        # Add movement rewards (closer/farther from goal)
        # These are already in base_reward and we keep them
        # Extract them: base_reward typically has +1/-0.5 components
        movement_reward = 0.0
        if -1 < base_reward < 5:  # Likely contains movement rewards
            movement_reward = base_reward
        
        reward += movement_reward
        
        return reward


class SimpleRewardModeWrapper(gym.Wrapper):
    """
    Simpler wrapper that just scales the base environment rewards.
    
    This is easier but less precise than reconstructing rewards.
    Use this if the complex wrapper has issues.
    """
    
    def __init__(self, env, mode='balanced'):
        super().__init__(env)
        self.mode = mode
        
        # Define multipliers for each component
        if mode == 'safe':
            self.collision_scale = 5.0      # -10 → -50
            self.step_scale = 0.5           # -0.1 → -0.05
            self.wait_scale = 0.13          # -0.15 → -0.02
            self.success_scale = 1.5        # +100 → +150
            self.speed_bonus = 0.0
            
        elif mode == 'efficient':
            self.collision_scale = 0.5      # -10 → -5
            self.step_scale = 3.0           # -0.1 → -0.3
            self.wait_scale = 3.3           # -0.15 → -0.5
            self.success_scale = 1.0        # +100 → +100
            self.speed_bonus = 50.0
            
        else:  # balanced
            self.collision_scale = 1.0
            self.step_scale = 1.0
            self.wait_scale = 1.0
            self.success_scale = 1.0
            self.speed_bonus = 0.0
        
        print(f"Simple Reward Wrapper: {mode.upper()} mode")
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Scale reward components (rough approximation)
        modified_reward = reward
        
        # Apply scaling (this is approximate!)
        if reward < -5:  # Collision
            modified_reward = reward * self.collision_scale
        elif -1 < reward < 0:  # Step/wait penalties
            modified_reward = reward * self.step_scale
        elif reward > 50:  # Success bonus
            modified_reward = reward * self.success_scale
            
            # Add speed bonus for efficient mode
            if self.speed_bonus > 0 and done and self.env.step_count < 15:
                modified_reward += self.speed_bonus
        
        return obs, modified_reward, done, truncated, info


if __name__ == "__main__":
    # Test the wrapper
    from compact_junction_env import CompactJunctionEnv
    
    print("\n" + "="*70)
    print("TESTING REWARD MODE WRAPPER")
    print("="*70)
    
    # Test each mode
    for mode in ['safe', 'balanced', 'efficient']:
        print(f"\n\nMode: {mode}")
        
        env = CompactJunctionEnv()
        env = RewardModeWrapper(env, mode=mode)
        
        # Run a few steps
        obs, _ = env.reset()
        
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            print(f"  Step {i+1}: reward = {reward:.2f}, done = {done}")
            
            if done:
                break
        
        print()
    
    print("="*70)
    print("✓ Wrapper test complete!")
    print("="*70)
