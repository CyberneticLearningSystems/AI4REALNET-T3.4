import os
import sys
import random
import argparse
import numpy as np
from typing import Any, Dict, Union

import torch

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(f"[Runner] Source path: {src_path}")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.utils.file_utils import load_config_file
from src.utils.observation.obs_utils import calculate_state_size
from src.configs.EnvConfig import _resolve_env_config, BaseEnvConfig
from src.configs.ControllerConfigs import PPOControllerConfig
from src.algorithms.PPO.PPOLearner import PPOLearner


def train_ppo(controller_config: PPOControllerConfig, learner_config: Dict, env_config: BaseEnvConfig, device: str) -> None:
    learner = PPOLearner(controller_config=controller_config,
                         learner_config=learner_config,
                         env_config=env_config,
                         device=device)

    # --- DEBUG PATCH ---
    original_optimise = learner._optimise

    def debug_optimise(*args, **kwargs):
        losses = original_optimise(*args, **kwargs)

        # Print GAE stats
        raw_gae_mean, raw_gae_std, gae_mean, gae_std = learner._normalise_gaes()

        # Print losses
        print("\n=== DEBUG LOSSES ===")
        print(f"Policy loss: {np.mean(losses['policy_loss']):.6f}")
        print(f"Value loss: {np.mean(losses['value_loss']):.6f}")
        print(f"Total loss: {np.mean(losses['total_loss']):.6f}")

        # Print parameter norms
        actor_norm = torch.nn.utils.parameters_to_vector(learner.controller.actor_network.parameters()).norm().item()
        critic_norm = torch.nn.utils.parameters_to_vector(learner.controller.critic_network.parameters()).norm().item()
        encoder_norm = torch.nn.utils.parameters_to_vector(learner.controller.encoder_network.parameters()).norm().item()
        print("\n=== DEBUG PARAMETER NORMS ===")
        print(f"Actor: {actor_norm:.4f}, Critic: {critic_norm:.4f}, Encoder: {encoder_norm:.4f}\n")

        return losses

    learner._optimise = debug_optimise
    # --- END DEBUG PATCH ---

    learner.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a PPO agent')
    parser.add_argument('--config_path', type=str, default='src/configs/PPO_FNN.yaml', help='Path to the configuration file')
    parser.add_argument('--wandb_project', type=str, default='AI4REALNET-JM', help='Weights & Biases project name for logging')
    parser.add_argument('--wandb_entity', type=str, default='jamarti96-fhnw', help='Weights & Biases entity name for logging')
    parser.add_argument('--random_seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the training on (cpu or cuda)')
    args = parser.parse_args()

    # Load config file
    config = load_config_file(args.config_path)

    # prepare environment config
    if args.random_seed:
        config['environment_config']['random_seed'] = args.random_seed
    env_config = _resolve_env_config(config['environment_config'])

    # prepare controller config and setup parallelisation
    learner_config = config['learner_config']
    learner_config['wandb_project'] = args.wandb_project
    learner_config['wandb_entity'] = args.wandb_entity

    # prepare controller
    controller_config_dict = config['controller_config']
    env_type = getattr(env_config, 'env_type', 'flatland')
    if env_type == 'flatland':
        n_nodes, state_size = calculate_state_size(env_config.observation_builder_config['max_depth'])
        controller_config_dict['n_nodes'] = n_nodes
        controller_config_dict['state_size'] = state_size
    else:
        controller_config_dict['state_size'] = getattr(env_config, 'state_size')
        controller_config_dict['action_size'] = getattr(env_config, 'action_size')
        controller_config_dict['n_nodes'] = controller_config_dict.get('n_nodes', 1)
        controller_config_dict['n_features'] = controller_config_dict['state_size']
    controller_config = PPOControllerConfig(controller_config_dict)

    train_ppo(controller_config=controller_config,
              learner_config=learner_config,
              env_config=env_config,
              device=args.device)
