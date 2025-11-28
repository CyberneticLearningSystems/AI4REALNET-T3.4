'''
File name:              BasePPO.py
Developed For:          AI4REALNET T3.4
Author:                 Julia Usher
Created:                24/11/2025
Date last modified:     24/11/2025
Python Version:         3.10.11
'''
import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List
from argparse import ArgumentParser

# Add project root to Python path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import wandb

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
                                                   n_features=int(self.state_size/self.n_nodes),
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
        self.value_loss_coef = LEARNING_PARAMS['value_loss_coef']
        self.entropy_coef = LEARNING_PARAMS['entropy_coef']


        # initialise optimiser
        self.optimizer = torch.optim.Adam(self.controller.parameters(), lr=self.learning_rate)

        # setup weights and biases
        wandb.init(project='Minimal_PPO', entity='CLS-FHNW', config=LEARNING_PARAMS, reinit=True)
        wandb.run.define_metric('episodes/*', step_metric='episode')
        wandb.run.define_metric('train/*', step_metric='epoch')
        wandb.run.name = RUN_NAME


    def setup_environment(self, max_tree_depth: int = 2, malfunctions: bool = False) -> RailEnv:
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

    # MAIN TRAINING LOOP
    def train(self):
        completed_updates = 0
        update_steps = 0
        # until we have completed the required number of SGD updates

        # TODO: simplify this to one update per episode --> sample size will likely be far too small.
        while completed_updates < self.sgd_updates:
            # set rollout buffer
            self.rollout_buffer = RolloutBuffer(state_size=self.state_size, n_agents=self.flatland_environment.get_num_agents())

            # reset episode
            episode_steps = 0
            current_state_dict, _ = self.flatland_environment.reset()
            current_state_tensor = obs_dict_to_tensor(observation=current_state_dict,
                                                      obs_type='tree',
                                                      n_agents=self.flatland_environment.get_num_agents(), 
                                                      max_depth=self.observation_builder.max_depth,
                                                      n_nodes=self.n_nodes)
            # normalise expects a tensor of shape (batch_size, n_agents, state_size)
            current_state_tensor = self.normalisation.normalise(current_state_tensor.unsqueeze(0)).squeeze(0)

            while True: 
                episode_steps += 1
                # sample actions and calculate state values
                log_probs, state_values = self.controller(current_state_tensor)
                actions_tensor = torch.argmax(log_probs, dim=1) # (n_agents, 1) #! this is selecting, not sampling!!!
                actions_dict = {i: int(actions_tensor[i]) for i in range(actions_tensor.shape[0])}

                # step environment
                next_state_dict, rewards_dict, dones_dict, _ = self.flatland_environment.step(actions_dict)
                next_state_tensor = obs_dict_to_tensor(observation=next_state_dict,
                                                      obs_type='tree',
                                                      n_agents=self.flatland_environment.get_num_agents(), 
                                                      max_depth=self.observation_builder.max_depth,
                                                      n_nodes=self.n_nodes)
                next_state_tensor = self.normalisation.normalise(next_state_tensor.unsqueeze(0)).squeeze(0)
                next_state_values = self.controller(next_state_tensor)[1]  # (n_agents, 1)

                # convert rewards and dones to tensors
                rewards_tensor = torch.tensor([rewards_dict[i] for i in range(self.flatland_environment.get_num_agents())]).unsqueeze(1)  # (n_agents, 1)
                dones_tensor = torch.tensor([dones_dict[i] for i in range(self.flatland_environment.get_num_agents())]).unsqueeze(1).float()  # (n_agents, 1)

                # add transition to rollout buffer
                #! debugging print
                print(f'current_state_tensor shape: {current_state_tensor.shape},\n actions_tensor shape: {actions_tensor.shape},\n log_probs shape: {log_probs.shape},\n rewards_tensor shape: {rewards_tensor.shape},\n next_state_tensor shape: {next_state_tensor.shape},\n dones_tensor shape: {dones_tensor.shape}')
                self.rollout_buffer.add_transition(states=current_state_tensor, 
                                                   state_values=state_values, 
                                                   next_state_values=next_state_values, 
                                                   actions=actions_tensor, 
                                                   log_probs=log_probs, 
                                                   rewards=rewards_tensor, 
                                                   dones=dones_tensor)

                if all(dones_dict.values()):
                    agent_rewards = self.rollout_buffer.rewards.sum(dim=0) # sum over time steps, new shape (n_agents, 1)
                    print(f'Agent rewards: {agent_rewards}')
                    average_agent_reward = agent_rewards.mean().item()
                    wandb.log({'episode': completed_updates,
                    'episode/total_reward': agent_rewards.sum().item(),
                    'episode/average_reward': average_agent_reward,
                    'episode/episode_length': episode_steps,
                    'episode/completion': sum([dones_dict[agent] for agent in range(self.n_agents)]) / self.n_agents})
                    break
            
            # perform learning update
            losses = {
                'policy_loss': [],
                'value_loss': [],
                'total_loss': []
            }

            for iteration in range(self.sgd_updates):
                #! no minibatching because we're only using one episode
                gaes, value_targets = self.advantages_targets(state_values=self.rollout_buffer.state_values,
                                                        next_state_values=self.rollout_buffer.next_state_values,
                                                        rewards=self.rollout_buffer.rewards,
                                                        dones=self.rollout_buffer.dones
                                                        )
                value_targets = value_targets.view(-1, 1)
                self.rollout_buffer.stack_trajectories()

                # evaluate log probs and state values for all states and actions in the buffer with current controller
                new_log_probs, entropy, new_state_values = self.controller.evaluate(self.rollout_buffer.states.view(-1, self.state_size), self.rollout_buffer.actions.view(-1, 1))

                # calculate losses
                policy_loss = self.policy_loss(new_log_probs, self.rollout_buffer.log_probs)
                value_loss = self.value_loss(new_state_values, value_targets)
                total_loss: Tensor = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                # backpropagate losses
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # accumulate losses for logging
                losses['policy_loss'].append(policy_loss.item())
                losses['value_loss'].append(value_loss.item())
                losses['total_loss'].append(total_loss.item())

                # weights and biases logging
                wandb.log({
                'epoch': update_steps,
                'train/policy_loss': np.mean(losses['policy_loss']),
                'train/value_loss': np.mean(losses['value_loss']),
                'train/total_loss': np.mean(losses['total_loss']),
                # 'train/raw_gae_mean': self.rollout_buffer.gae.mean().item(),
                # 'train/raw_gae_std': self.rollout_buffer.gae.std().item(),
                'train/mean_entropy': entropy.mean().item()
                })
                update_steps += 1
            completed_updates += 1
            

    # LOSS FUNCTIONS
    def value_loss(self, new_state_values: Tensor, value_targets: Tensor) -> Tensor:
        """ 
        Simple MSE value loss function, copied from src/algorithms/loss.py "value_loss" function 

        Parameters:
            - new_state_values: Tensor of shape (episode_length * n_agents, 1)
            - targets: Tensor of shape (episode_length * n_agents, 1)
        """
        return F.mse_loss(new_state_values, value_targets)
    

    def policy_loss(self, new_log_prob, old_log_prob):
        """ 
        PPO clipped policy loss function, copied from src/algorithms/loss.py "policy_loss" function 

        Parameters:
            - new_log_prob: Tensor of shape (episode_length, n_agents, 1)
            - old_log_prob: Tensor of shape (episode_length, n_agents, 1)
        """
        # TODO: check dimensions
        unclipped_ratio: Tensor = torch.exp(new_log_prob - old_log_prob)
        clipped_ratio: Tensor = torch.clamp(unclipped_ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        actor_loss: Tensor = -torch.min(clipped_ratio * self.rollout_buffer.gaes, unclipped_ratio * self.rollout_buffer.gaes).mean()
        return actor_loss
    

    # GENERALISED ADVANTAGE CALCULATION
    def advantages_targets(self, state_values: Tensor, next_state_values: Tensor, rewards: Tensor, dones: Tensor) -> Tuple[Tensor, Tensor]:
        """ 
        Calculate Generalized Advantage Estimates (GAE) and the value targets -> weighted sum of temporal differences, standard for PPO
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
        value_targets = torch.zeros_like(rewards)  # (batchsize, n_agents, 1)
        gaes = torch.zeros_like(rewards)  # (batchsize, n_agents, 1)

        for agent_index in range(rewards.shape[1]):  # iterate over agents
            gae = 0.0
            length = rewards.shape[0]

            for t in reversed(range(length)):  # iterate backwards over time steps
                if t == length - 1:
                    next_non_terminal = 0.0
                    next_state_value = 0.0 # we don't know the next state, this is a bootstrap
                else: 
                    if dones[t, agent_index]: 
                        next_non_terminal = 0.0
                    else: 
                        next_non_terminal = 1.0

                    next_state_value = next_state_values[t, agent_index] * next_non_terminal

                # TD error
                delta = rewards[t, agent_index] + self.gamma * next_state_value - state_values[t, agent_index]

                # GAE recursive calculation
                gae = delta + self.gamma * self.lam * next_non_terminal * gae

                # fill into tensors
                gaes[t, agent_index] = gae
                value_targets[t, agent_index] = gae + state_values[t, agent_index]

        return gaes, value_targets


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
        self.states: List[Tensor]= []
        self.state_values: List[Tensor]= []
        self.next_state_values: List[Tensor]= []
        self.actions: List[Tensor]= []
        self.log_probs: List[Tensor]= []
        self.rewards: List[Tensor]= []
        self.dones: List[Tensor]= []
        self.gaes: List[Tensor]= []

    def add_transition(self, states: Tensor, state_values: Tensor, next_state_values: Tensor, actions: Tensor, log_probs: Tensor, rewards: Tensor, dones: Tensor) -> None:
        """ 
        Add a new transition to the buffer. 
            - Values come in the shape (n_agents, feature_size) and are resizes to (1, n_agents, feature_size) before concatenation at dimension 0
        """
        self.states.append(states.unsqueeze(0))
        self.state_values.append(state_values.unsqueeze(0))
        self.next_state_values.append(next_state_values.unsqueeze(0))
        self.actions.append(actions.unsqueeze(0))
        self.log_probs.append(log_probs.unsqueeze(0))
        self.rewards.append(rewards.unsqueeze(0))
        self.dones.append(dones.unsqueeze(0))
        self.episode_steps += 1

    def add_gaes(self, gaes: Tensor) -> None:
        """ 
        Add the calculated GAEs to the buffer. Expects size (episode_steps, n_agents, 1).
        """
        self.gae: Tensor = gaes

    def stack_trajectories(self) -> None:
        """ 
        Stack all episodes in the buffer into a single tensor for each component.
        """
        self.states = torch.stack(self.states)  # (episode_steps, n_agents, state_size)
        self.state_values = torch.stack(self.state_values)  # (episode_steps, n_agents, 1)
        self.next_state_values = torch.stack(self.next_state_values)  # (episode_steps, n_agents, 1)
        self.actions = torch.stack(self.actions)  # (episode_steps, n_agents, 1)
        self.log_probs = torch.stack(self.log_probs)  # (episode_steps, n_agents, 1)
        self.rewards = torch.stack(self.rewards)  # (episode_steps, n_agents, 1)
        self.dones = torch.stack(self.dones)  # (episode_steps, n_agents, 1)
        self.gaes = self.gae.view(-1, 1)  # (episode_steps * n_agents, 1)


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
        log_probs = F.log_softmax(action_logits, dim=-1)
        return action_logits, state_value
        

    def evaluate(self, states: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor]:
        """ 
        Evaluate the log probabilities and state values for given states and actions. Copied from src/algorithms/PPO/PPOLearner.py "_evaluate" function lines 392-404
        
        Parameters:
            - states: Tensor of shape (n_agents, state_size)
            - actions: Tensor of shape (n_agents, 1)

        Returns:
            - log_prob: Tensor of shape (n_agents, 1)
            - state_values: Tensor of shape (n_agents, 1)
        """
        encoded_states = self.encoder_network(states) # (n_agents, encoder_output_size)
        action_logits = self.actor_network(encoded_states) # (n_agents, action_size)
        state_values = self.critic_network(encoded_states) # (n_agents, 1)

        action_distribution = torch.distributions.Categorical(logits=action_logits)
        entropy = action_distribution.entropy().mean() # (1,)
        log_probs = action_distribution.log_prob(actions)  # (batch_size, 1)

        return log_probs, entropy, state_values


if __name__ == "__main__":
    ppo = BasePPO()
    ppo.train()  

