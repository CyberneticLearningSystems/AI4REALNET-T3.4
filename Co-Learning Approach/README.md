# Co-Learning Approach
This approach focuses on multi-agent reinforcement learning with human-in-the-loop decision support. 

Python Version: 3.10

# Install dependencies
pip install -r requirements.txt

# Training
To train new models:
python train_multimode.py --all --timesteps 500000 (trains all 3 models: safe, balanced, efficient)
python train_multimode.py --mode efficient --timesteps 500000
python train_multimode.py --mode safe --timesteps 500000
python train_multimode.py --mode balanced ->(if no timesteps are defined the predefined values will be used)

# HMI:co-learning App with human decision
python human_in_loop_compact.py --model "Balanced:models/compact_3x5_final.zip" --model "Safe:models/safe_final.zip" --model "Efficient:models/efficient_final.zip" (loads all 3 predefined models)

It is also possible to just train one or two models:
python human_in_loop_compact.py --safe models/safe.zip --balanced models/compact_3x5_final.zip

# Test of environments without training
python compact_junction_env.py

# Short Overview
1. compact_junction_env.py - the environment
The "Game/Grid": 3×5 Grids, 3 trains, 2 junctions. Includes complete movement logic, collision detection, and basic rewards.
Reward modes:
- Safe: high success reward, moderate collision penalty
- Balanced: standard rewards
- Efficient: speed bonus, lower collision penalty

2. conflict_detector_compact.py - conflict detection module
Look straight ahead for each move (5 steps) and check if paths intersect. Reports: "Conflict at position (1,2)!"

3. corridor_visualization_widget.py - the graphic
Draws the environment using Matplotlib: tracks (black), crossings (yellow), trains (colored circles), conflicts (red circles).

4. human_in_loop_compact.py - the GUI application
The main application: Loads models, displays the environment, pauses in case of conflicts, and lets humans decide.

5. reward_mode_wrapper.py - reward configuration
Handles different reward modes (safe, balanced, efficient) and applies them during training and execution.

6. train_multimode.py - the training script
Trains and saves the 3 PPO models. Can also evaluate models (statistics over 100+ episodes).
