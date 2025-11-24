'''
File name:              BasePPO.py
Developed For:          AI4REALNET T3.4
Author:                 Julia Usher
Created:                24/11/2025
Date last modified:     24/11/2025
Python Version:         3.10.11
'''
import os
from typing import Dict, Any, Tuple
from argparse import ArgumentParser

# torch imports
import torch
from torch import nn, Tensor
import torch.nn.functional as F

# Observation Formatting and Normalisation
from src.utils.observation.normalisation import FlatlandNormalisation
from src.utils.observation.obs_utils import obs_dict_to_tensor, calculate_state_size

# Flatland Imports
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv


# RUN SETTINGS HERE
RUN_NAME = 'BasePPO_Run'
LEARNING_PARAMS = {
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'lambda': 0.95,
    'clip_epsilon': 0.2,
    'sgd_updates': 10,
    'samples_per_update': 2048,
    'sgd_batch_size': 64,
    'entropy_coef': 0.01,
    'value_loss_coef': 0.5,
    }


class BasePPO(): 
    def __init__(self):
        # setup flatland environment
        self.flatland_environment: RailEnv = self.setup_environment()

        # determine observation parameters
        self.state_size: int = 0
        self.n_nodes: int = 0
        self.n_nodes, self.state_size = calculate_state_size(max_depth=self.observation_builder.max_depth)

        # setup controller network
        self.controller = ControllerNetwork(
            state_size = self.state_size,
            encoder_hidden_size = 128,
            encoder_output_size = 64,
            action_size = self.flatland_environment.action_space[0], # this is [5] for 5 actions (no clue why it's programmed this way)
        )

        # setup normalisation
        self.normalisation = FlatlandNormalisation(n_nodes=self.n_nodes, 
                                                   state_size=self.state_size,
                                                   n_agents=self.flatland_environment.get_num_agents(),
                                                   env_size=(self.flatland_environment.width, self.flatland_environment.height)
                                                   )
        
        # load learning parameters
        self.gamma = LEARNING_PARAMS['gamma']
        self.clip_epsilon = LEARNING_PARAMS['clip_epsilon'] 
        self.learning_rate = LEARNING_PARAMS['learning_rate']
        self.sgd_updates = LEARNING_PARAMS['sgd_updates']
        self.samples_per_update = LEARNING_PARAMS['samples_per_update']
        self.sgd_batch_size = LEARNING_PARAMS['sgd_batch_size']
        self.lam = LEARNING_PARAMS['lambda']

        # initialise optimiser
        self.optimizer = torch.optim.Adam(self.controller.parameters(), lr=self.learning_rate)


    def setup_environment(self, observation_type: str = 'tree', max_tree_depth: int = 2, malfunctions: bool = False) -> RailEnv:
        """ Setup a small Flatland RailEnv environment. - copied from src/environments/env_small.py """

        # creates a tree observation generator 
        self.observation_builder: TreeObsForRailEnv = TreeObsForRailEnv(max_depth=max_tree_depth, predictor=ShortestPathPredictorForRailEnv())

        # creates the generator that will generate the rails (i.e., network topology) for the environment
        rail_generator = sparse_rail_generator(
            max_num_cities=4,
            max_rails_between_cities=2,
            max_rail_pairs_in_city=2,
            grid_mode=False
        )
        
        # speed: ratio of trains running at this speed
        speed_ratio_map = {1.: 0.7,
                           0.5: 0.3}
        
        # creates the lines for the environment based on the speed ratio map
        line_generator = sparse_line_generator(speed_ratio_map)

        # Sets the parameters for malfunctions and then creates the malfunction generator for the flatland environment
        if malfunctions:
            stochastic_malfunctions = MalfunctionParameters(malfunction_rate=1/1000,
                                                            min_duration=20,
                                                            max_duration=50)
            malfunction_generator = ParamMalfunctionGen(stochastic_malfunctions)
        else:
            malfunction_generator = None
        
        # Create the Flatland RailEnv environment using the previously defined parameters
        return RailEnv(width=35,
                    height=28,
                    number_of_agents=8,
                    rail_generator=rail_generator,
                    line_generator=line_generator,
                    malfunction_generator=malfunction_generator,
                    obs_builder_object=self.observation_builder
                    )

    def train(self):
        completed_updates = 0
        # until we have completed the required number of SGD updates

        # TODO: simplify this to one update per episode --> sample size will likely be far too small.
        while completed_updates < self.sgd_updates:
            # set rollout buffer
            rollout_buffer = RolloutBuffer(state_size=self.state_size, n_agents=self.flatland_environment.get_num_agents())

            # gather rollouts
            if rollout_buffer.episode_steps < self.samples_per_update:
                obs_dict, infos = self.flatland_environment.reset()
                # get observations from environment and convert to tensor
                obs_tensor = obs_dict_to_tensor(obs_dict, self.flatland_environment.get_num_agents(), self.state_size)
                # normalise observations
                norm_obs = self.normalisation.normalise(obs_tensor)


    def value_loss(self, predictions, targets):
        """ Simple MSE value loss function, copied from src/algorithms/loss.py "value_loss" function """
        return F.mse_loss(predictions, targets)
    
    def policy_loss(self, gae, new_log_prob, old_log_prob):
        """ PPO clipped policy loss function, copied from src/algorithms/loss.py "policy_loss" function """
        unclipped_ratio: Tensor = torch.exp(new_log_prob - old_log_prob)
        clipped_ratio: Tensor = torch.clamp(unclipped_ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        actor_loss: Tensor = -torch.min(clipped_ratio * gae, unclipped_ratio * gae).mean()
        return actor_loss

    def gaes(self, states: Tensor, next_states: Tensor, rewards: Tensor, dones: Tensor) -> Tensor:
        """ 
        Calculate Generalized Advantage Estimates (GAE) -> weighted sum of temporal differences, standard for PPO
        Paper: https://arxiv.org/abs/1506.02438

        The temporal difference residuals are calculated as: 
            delta_t = r_t + gamma * V(s_{t+1) - V(s_t)})

        The GAE then computes the weighted sum of these TD-errors recursively as:
            A_t = sum(l=0 to infinity) (gamma * lambda)^l * delta_{t+l}

            where gamma is the discount factor and lambda is a hyperparameter that controls the bias-variance trade-off. 
                lambda=1 is monte carlo - high variance, low bias
                lambda=0 is 1-step TD - low variance, high bias

        Outputs a tensor of shape (episode_length, n_agents, 1)
        """
        gaes = torch.zeros_like(rewards)  # (batchsize, n_agents, 1)
        for episode_step in range(rewards.shape[0]):  # iterate over episodes in batch
            for agent_index in range(rewards.shape[1]):  # iterate over agents
                # calculate GAE over the whole episode recursively
                _, state_values = self.controller(states[episode_step, agent_index, :])
                advantage = 0.0
                for t in reversed(range(rewards.shape[2])):  # iterate backwards over time steps
                    delta = rewards[episode_step, agent_index, t] + self.gamma * next_states[episode_step, agent_index, t] * (1 - dones[episode_step, agent_index, t]) - state_values[t]
                    advantage = delta + self.gamma * self.lam * (1- dones[episode_step, agent_index, t]) * advantage
                    gaes[episode_step, agent_index, t] = advantage
        return gaes


    def save_model(self) -> None:
        """Persist the current controller parameters to disk."""
        savepath = os.path.join('models', f'{self.run_name}')
        os.makedirs(savepath, exist_ok=True)
        torch.save(self.controller.actor_network.state_dict(), os.path.join(savepath, 'actor.pth'))
        torch.save(self.controller.critic_network.state_dict(), os.path.join(savepath, 'critic.pth'))
        torch.save(self.controller.encoder_network.state_dict(), os.path.join(savepath, 'encoder.pth'))
        print(f'Model parameters saved to {savepath}')


class RolloutBuffer(): 
    """
    Simplified from MultiAgentRolloutBuffer in src/algorithms/BasePPO/MultiAgentPPO.py
    """
    def __init__(self, state_size: int, n_agents: int):
        # Number of steps to track buffer size
        self.episode_steps: int = 0
        
        # Initialise empty tensors for each component of the buffer
        #! This structure doesn't allow for episodes of varying lengths 
        self.state: torch.Tensor = torch.Tensor(size=(0, n_agents, state_size))
        self.next_state: torch.Tensor = torch.Tensor(size=(0, n_agents, state_size))
        self.action: torch.Tensor = torch.Tensor(size=(0, n_agents, 1))
        self.log_prob: torch.Tensor = torch.Tensor(size=(0, n_agents, 1))
        self.reward: torch.Tensor = torch.Tensor(size=(0, n_agents, 1))
        self.done: torch.Tensor = torch.Tensor(size=(0, n_agents, 1))

    def add_transition(self, state: Tensor, action: Tensor, log_prob: Tensor, reward: Tensor, next_state: Tensor, done: Tensor) -> None:
        """ 
        Add a new transition to the buffer. 
            - Values come in the shape (n_agents, feature_size) and are resizes to (1, n_agents, feature_size) before concatenation at dimension 0
        """
        self.state = torch.cat((self.state, state.unsqueeze(0)), dim=0)
        self.action = torch.cat((self.action, action.unsqueeze(0)), dim=0)
        self.log_prob = torch.cat((self.log_prob, log_prob.unsqueeze(0)), dim=0)
        self.reward = torch.cat((self.reward, reward.unsqueeze(0)), dim=0)
        self.next_state = torch.cat((self.next_state, next_state.unsqueeze(0)), dim=0)
        self.done = torch.cat((self.done, done.unsqueeze(0)), dim=0)
        self.episode_steps += 1

    def add_gaes(self, gaes: Tensor) -> None:
        """ 
        Add the calculated GAEs to the buffer. Expects size (episode_steps, n_agents, 1).
        """
        self.gae = gaes

    def stack_episodes(self) -> None:
        """ 
        Stack all episodes in the buffer into a single tensor for each component.
        """
        pass


class ControllerNetwork(nn.Module):
    """ 
    Controller Network for PPO algorithm with encoder, actor, and critic networks. 
    Copied and simplified from src/controllers/PPOController.py and src/networks/FeedForwardNN.py        
    """
    def __init__(self, state_size: int, encoder_hidden_size: int, encoder_output_size: int, action_size: int):
        super(ControllerNetwork, self).__init__()
        # Define encoder network
        self.encoder_network = nn.Sequential(
            nn.Linear(in_features = state_size, out_features = encoder_hidden_size),
            nn.ReLU(),
            nn.Linear(in_features = encoder_hidden_size, out_features = encoder_output_size),
            nn.ReLU()
        )
        # Define actor network
        self.actor_network = nn.Sequential(
            nn.Linear(in_features = encoder_output_size, out_features = action_size),
        )

        # Define critic network
        self.critic_network = nn.Sequential(
            nn.Linear(in_features=encoder_output_size, out_features= 1)
        )

    def forward(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        """ Forward pass through the network. """
        encoded_state = self.encoder_network(state)
        action_logits = self.actor_network(encoded_state)
        state_value = self.critic_network(encoded_state)
        return action_logits, state_value


if __name__ == "__main__":
    ppo = BasePPO()
    ppo.train()  

