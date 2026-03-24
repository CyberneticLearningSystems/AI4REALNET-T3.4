"""
Train Multiple Models with Different Reward Modes
==================================================
Trains 3 separate models with different reward structures:
- Safe Mode: Prioritizes collision avoidance
- Efficient Mode: Prioritizes speed
- Balanced Mode: Balance of both

Usage:
    # Train single mode
    python train_multimode.py --mode safe
    
    # Train all 3 modes sequentially
    python train_multimode.py --all
    
    # Custom timesteps
    python train_multimode.py --mode efficient --timesteps 500000
"""

import argparse
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from compact_junction_env import CompactJunctionEnv
from reward_mode_wrapper import RewardModeWrapper

# Try to import wandb components (optional)
try:
    import wandb
    from wandb_callback import WandbSimpleJunctionCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


def train_mode(mode, total_timesteps=150_000, use_wandb=True):
    """
    Train a single model with specified mode.
    
    Args:
        mode: 'safe', 'efficient', or 'balanced'
        total_timesteps: Number of training steps
        use_wandb: Whether to use Weights & Biases logging
    """
    
    print(f"\n{'='*70}")
    print(f"TRAINING {mode.upper()} MODE")
    print(f"{'='*70}\n")
    
    # Create base environment
    env = CompactJunctionEnv()
    
    # Wrap with reward mode
    env = RewardModeWrapper(env, mode=mode)
    
    # Setup wandb
    if use_wandb and WANDB_AVAILABLE:
        run_name = f"{mode}_ppo"
        wandb.init(
            project="compact-3x5-multimode",
            name=run_name,
            config={
                "mode": mode,
                "total_timesteps": total_timesteps,
                "algorithm": "PPO",
                "env": "CompactJunction3x5",
                "n_agents": 3,
            },
            sync_tensorboard=False,  # Disabled - not needed
            monitor_gym=False,       # Disabled - not needed
        )
    elif use_wandb and not WANDB_AVAILABLE:
        print("Warning: wandb requested but not available. Continuing without wandb.")
        use_wandb = False
    
    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log=None,  # Disabled - not needed
    )
    
    # Setup callbacks
    callbacks = []
    
    # Checkpoint callback (save every 10k steps)
    checkpoint_dir = Path(f"models/{mode}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=str(checkpoint_dir),
        name_prefix=f"{mode}_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    callbacks.append(checkpoint_callback)
    
    # WandB callback
    if use_wandb and WANDB_AVAILABLE:
        wandb_callback = WandbSimpleJunctionCallback(
            log_every=1,
            verbose=1,
        )
        callbacks.append(wandb_callback)
    
    # Train
    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    print(f"{'='*70}\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    
    # Save final model
    final_path = Path(f"models/{mode}_final.zip")
    model.save(str(final_path))
    
    print(f"\n{'='*70}")
    print(f"✓ {mode.upper()} MODE TRAINING COMPLETE!")
    print(f"Final model saved to: {final_path}")
    print(f"{'='*70}\n")
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    return model


def train_all_modes(total_timesteps=150_000, use_wandb=True):
    """Train all 3 modes sequentially"""
    
    print(f"\n{'#'*70}")
    print(f"TRAINING ALL 3 MODES")
    print(f"Total timesteps per mode: {total_timesteps:,}")
    print(f"{'#'*70}\n")
    
    modes = ['balanced', 'safe', 'efficient']
    
    for i, mode in enumerate(modes, 1):
        print(f"\n{'#'*70}")
        print(f"MODE {i}/3: {mode.upper()}")
        print(f"{'#'*70}\n")
        
        train_mode(mode, total_timesteps, use_wandb)
        
        print(f"\n✓ Completed {i}/{len(modes)} modes\n")
    
    print(f"\n{'#'*70}")
    print(f"🎉 ALL 3 MODES TRAINED SUCCESSFULLY!")
    print(f"{'#'*70}")
    print(f"\nModels saved:")
    print(f"  • models/safe_final.zip")
    print(f"  • models/balanced_final.zip")
    print(f"  • models/efficient_final.zip")
    print(f"\nTo use in human-in-the-loop:")
    print(f"  python human_in_loop_compact.py \\")
    print(f"    --safe models/safe_final.zip \\")
    print(f"    --balanced models/balanced_final.zip \\")
    print(f"    --efficient models/efficient_final.zip")
    print(f"{'#'*70}\n")


def evaluate_mode(mode, n_episodes=100):
    """Evaluate a trained model"""
    
    print(f"\n{'='*70}")
    print(f"EVALUATING {mode.upper()} MODE")
    print(f"{'='*70}\n")
    
    # Load model
    model_path = Path(f"models/{mode}_final.zip")
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    model = PPO.load(str(model_path))
    
    # Create environment
    env = CompactJunctionEnv()
    env = RewardModeWrapper(env, mode=mode)
    
    # Statistics
    successes = 0
    collisions = 0
    total_steps = []
    total_rewards = []
    
    print(f"Running {n_episodes} episodes...")
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done and env.step_count < env.max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
        
        # Check results
        success = all(env.agents_done)
        collision = episode_reward < -50  # Heuristic
        
        if success:
            successes += 1
        if collision:
            collisions += 1
        
        total_steps.append(env.step_count)
        total_rewards.append(episode_reward)
        
        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{n_episodes}: "
                  f"Success={successes}/{ep+1} ({100*successes/(ep+1):.1f}%), "
                  f"Avg Steps={sum(total_steps)/(ep+1):.1f}")
    
    # Print results
    print(f"\n{'='*70}")
    print(f"RESULTS FOR {mode.upper()} MODE")
    print(f"{'='*70}")
    print(f"Episodes:        {n_episodes}")
    print(f"Success Rate:    {successes}/{n_episodes} ({100*successes/n_episodes:.1f}%)")
    print(f"Collision Rate:  {collisions}/{n_episodes} ({100*collisions/n_episodes:.1f}%)")
    print(f"Avg Steps:       {sum(total_steps)/n_episodes:.1f}")
    print(f"Avg Reward:      {sum(total_rewards)/n_episodes:.1f}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Train models with different reward modes')
    
    parser.add_argument('--mode', type=str, choices=['safe', 'efficient', 'balanced'],
                       help='Training mode (safe/efficient/balanced)')
    parser.add_argument('--all', action='store_true',
                       help='Train all 3 modes sequentially')
    parser.add_argument('--eval', action='store_true',
                       help='Evaluate trained model instead of training')
    parser.add_argument('--timesteps', type=int, default=300_000,
                       help='Total training timesteps (default: 300,000)')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    parser.add_argument('--eval-episodes', type=int, default=100,
                       help='Number of evaluation episodes (default: 100)')
    
    args = parser.parse_args()
    
    use_wandb = not args.no_wandb
    
    if args.eval:
        # Evaluation mode
        if args.mode:
            evaluate_mode(args.mode, args.eval_episodes)
        else:
            # Evaluate all modes
            for mode in ['safe', 'balanced', 'efficient']:
                evaluate_mode(mode, args.eval_episodes)
    
    elif args.all:
        # Train all 3 modes
        train_all_modes(args.timesteps, use_wandb)
    
    elif args.mode:
        # Train single mode
        train_mode(args.mode, args.timesteps, use_wandb)
    
    else:
        print("Error: Must specify either --mode, --all, or --eval")
        parser.print_help()


if __name__ == "__main__":
    main()