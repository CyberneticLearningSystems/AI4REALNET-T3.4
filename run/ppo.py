import os
import sys
import random
import argparse
import numpy as np
from typing import Any, Dict, Union

import torch

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(src_path)
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
    learner.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a PPO agent')
    parser.add_argument('--config_path', type=str, default='src/configs/PPO_FNN.yaml', help='Path to the configuration file')
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


    train_ppo(controller_config = controller_config,
              learner_config = learner_config,
              env_config = env_config, 
              device = args.device)
