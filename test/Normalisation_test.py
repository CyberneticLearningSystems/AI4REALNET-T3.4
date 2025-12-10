import unittest
import torch
from typing import List, Dict

# ObservationTest functions / classes
from src.utils.observation.RunningMeanStd import RunningMeanStd, FeatureRunningMeanStd
from src.utils.observation.normalisation import Normalisation, FlatlandNormalisation

# PPOLearner normalisation classes
from src.algorithms.PPO.PPOLearner import PPOLearner
from src.utils.file_utils import load_config_file
from src.utils.observation.obs_utils import calculate_state_size
from src.configs.EnvConfig import _resolve_env_config, BaseEnvConfig
from src.configs.ControllerConfigs import PPOControllerConfig
from src.memory.MultiAgentRolloutBuffer import MultiAgentRolloutBuffer


class ObservationTest(unittest.TestCase):
    def setUp(self):
        self.single_feature_batch = torch.tensor([[1.0, 2.0, 3.0],
                                                  [4.0, 5.0, 6.0]], dtype=torch.float32)
        self.multi_feature_batch = torch.tensor([[1.0, 2.0, 3.0],
                                                 [4.0, 5.0, 6.0]], dtype=torch.float32)
        self.scalar_batch = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        self.tree_observation = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                                               2.0, 4.0, 6.0, 8.0, 10.0]], dtype=torch.float32)


    def test_value_normalisation(self):
        rms = RunningMeanStd(size=1)
        rms.update_batch(self.single_feature_batch)

        self.assertEqual(rms.count, self.single_feature_batch.size(0))

        expected_mean = torch.mean(self.single_feature_batch)
        self.assertAlmostEqual(rms.mean.item(), expected_mean.item(), places=6)

        expected_M2 = ((self.single_feature_batch - expected_mean) ** 2).sum()
        expected_var = expected_M2 / (self.single_feature_batch.size(0) + rms.eps)
        self.assertAlmostEqual(rms.var.item(), expected_var.item(), places=6)


    def test_feature_normalisation(self):
        feature_rms = FeatureRunningMeanStd(n_features=self.multi_feature_batch.size(1))
        feature_rms.update_batch(self.multi_feature_batch)

        self.assertEqual(feature_rms.count, self.multi_feature_batch.size(0))

        expected_mean = torch.mean(self.multi_feature_batch, dim=0)
        self.assertTrue(torch.allclose(feature_rms.mean, expected_mean, atol=1e-6))

        expected_M2 = ((self.multi_feature_batch - expected_mean) ** 2).sum(dim=0)
        expected_var = expected_M2 / (self.multi_feature_batch.size(0) + feature_rms.eps)
        self.assertTrue(torch.allclose(feature_rms.var, expected_var, atol=1e-6))


    def test_tensor_normalisation(self):
        normaliser = Normalisation(eps=1e-8, clip=False)
        normaliser.update_metrics(self.scalar_batch)

        normalised = normaliser.normalise(self.scalar_batch, clip=False)

        expected_mean = self.scalar_batch.mean()
        expected_var = ((self.scalar_batch - expected_mean) ** 2).sum() / (self.scalar_batch.size(0) + normaliser.rms.eps)
        expected_std = torch.sqrt(expected_var + normaliser.eps)
        expected = (self.scalar_batch - expected_mean) / (expected_std + normaliser.eps)

        self.assertTrue(torch.allclose(normalised, expected, atol=1e-6))


    def test_tree_observation_normalisation(self):
        flat_norm = FlatlandNormalisation(n_nodes=1, n_features=12, n_agents=2, eps=1e-8, clip=False)
        normalised = flat_norm.normalise(self.tree_observation.clone(), clip=False)

        self.assertEqual(normalised.shape, self.tree_observation.shape)

        distance_mean = flat_norm.distance_rms.mean
        distance_std = flat_norm.distance_rms.std + flat_norm.eps
        expected_distance = (self.tree_observation[:, :7] - distance_mean) / distance_std

        expected_agent_counts = self.tree_observation[:, 7:10] / flat_norm.n_agents
        expected_speed = self.tree_observation[:, 10:11]
        expected_last_agent_count = self.tree_observation[:, 11:12] / flat_norm.n_agents

        expected = torch.cat(
            [expected_distance, expected_agent_counts, expected_speed, expected_last_agent_count], dim=1
        )

        self.assertTrue(torch.allclose(normalised, expected, atol=1e-6))


class PPOLearnerNormalisationTest(unittest.TestCase):
    def setUp(self):
        # set and load config file
        config_path = 'src/configs/PPO_FNN.yaml'
        config = load_config_file(config_path)

        # prepare environment config
        env_config = _resolve_env_config(config['environment_config'])

        # prepare controller config and setup parallelisation
        learner_config = config['learner_config']
        learner_config['use_wandb'] = False  # Disable wandb for testing

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

        self.learner = PPOLearner(controller_config=controller_config,
                             learner_config=learner_config,
                             env_config=env_config,
                             device='cpu')
        
        # test tensor for episodes of length 9
        self.batch = torch.tensor([[10.0, 20.0, 30.0, 
                                    40.0, 50.0, 60.0, 
                                    70.0, 80.0, 90.0]], dtype=torch.float32)
        

        self.batch_average = torch.mean(self.batch)
        self.batch_std = torch.std(self.batch).clamp(min=1e-8)

        self.batch_normalised = (self.batch - self.batch_average) / self.batch_std

        self.episode_dict: Dict = {
            'states': [],
            'state_values': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'next_states': [],
            'next_state_values': [],
            'dones': [],
            'extras': {},
            'gaes': [[] for _ in range(2)],
            'episode_length': [0 for _ in range(2)],
            'total_episode_reward': [0.0 for _ in range(2)],
            'per_agent_average_reward': [0.0 for _ in range(2)],
            'average_episode_length': 0,
            'total_reward': 0.0,
            'average_episode_reward': 0.0
        }
        self.value_keys = ['states', 'state_values', 'actions', 'log_probs', 'rewards',
                           'next_states', 'next_state_values', 'dones', 'gaes']

    def create_rollout(self) -> MultiAgentRolloutBuffer:
        # GOAL: create a rollout with two episodes, two agents, each with the same batch data of length 9
        rollout = MultiAgentRolloutBuffer(n_agents=2)
        
        rollout.episodes: List[Dict] = [self.episode_dict.copy() for _ in range(2)]  # create two episodes based on,
        
        for episode in range(2):
            for key in self.value_keys:
                rollout.episodes[episode][key] = [self.batch.clone(),
                                                  self.batch.clone()]
                rollout.episodes[episode][key] = [self.batch.clone(),
                                                  self.batch.clone()]

        stacked_gaes = torch.cat([torch.cat(rollout.episodes[episode]['gaes']) for episode in range(2)], dim=0)      
        self.rollout_mean = stacked_gaes.mean()
        self.rollout_std = stacked_gaes.std().clamp(min=1e-8)
    
        return rollout
    
    def create_transition(self) -> MultiAgentRolloutBuffer:
        rollout = MultiAgentRolloutBuffer(n_agents=2)
        self.rewards = torch.cat([self.batch.clone() for _ in range(4)], dim=0)
        self.rewards_mean = self.rewards.mean()
        self.rewards_std = self.rewards.std().clamp(min=1e-8)
        rollout.transitions = {'rewards': self.rewards}
        self.normalised_rewards = (self.rewards - self.rewards_mean) / self.rewards_std
        return rollout
    

    def test_gae_normalisation(self):
        rollout = self.create_rollout()
        self.learner.rollout = rollout
        self.learner.n_agents = 2

        # checking content
        test_gaes = self.learner.rollout.episodes[0]['gaes'][0]  # access to gaes of agent 0 in episode 0
        verfication_gaes = rollout.episodes[0]['gaes'][0]
        self.assertTrue(torch.equal(test_gaes, verfication_gaes))

        # Update normalisation metrics with the batch
        raw_gae_mean, raw_gae_std, gae_mean, gae_std = self.learner._normalise_gaes()

        # Result printout for verification
        print(f"\n{'Metric':<20} {'Calculated':<15} {'Target':<15} {'Difference':<15}")
        print("-" * 65)
        print(f"{'Raw GAE Mean':<20} {raw_gae_mean:<15.5f} {self.rollout_mean.item():<15.5f} {abs(raw_gae_mean - self.rollout_mean.item()):<15.5f}")
        print(f"{'Raw GAE Std':<20} {raw_gae_std:<15.5f} {self.rollout_std.item():<15.5f} {abs(raw_gae_std - self.rollout_std.item()):<15.5f}")
        print(f"{'GAE Mean':<20} {gae_mean:<15.5f} {0.0:<15.5f} {abs(gae_mean - 0.0):<15.5f}")
        print(f"{'GAE Std':<20} {gae_std:<15.5f} {1.0:<15.5f} {abs(gae_std - 1.0):<15.5f}\n")

        # extract normalised gaes for further verification
        normalised_gaes = self.learner.rollout.episodes[0]['gaes'][0]  # access to gaes of agent 0 in episode 0
        print(f"{'Normalised GAEs':<20} {normalised_gaes}")
        print(f"{'Expected Normalised':<20} {self.batch_normalised}\n")
        

        # Assertions
        self.assertAlmostEqual(raw_gae_mean, self.rollout_mean.item(), places=3)
        self.assertAlmostEqual(raw_gae_std, self.rollout_std.item(), places=3)
        self.assertAlmostEqual(gae_mean, 0.0, places=3)
        self.assertAlmostEqual(gae_std, 1.0, places=3)

    def test_reward_normalisation(self):
        rollout = self.create_transition()
        self.learner.rollout = rollout
        # self.learner.n_agents = 2

        # compare initial rewards
        reward_transitions: torch.Tensor = self.learner.rollout.transitions['rewards']
        reward_raw_mean = reward_transitions.mean()
        reward_raw_std = reward_transitions.std().clamp(min=1e-8)
        self.assertEqual(reward_raw_mean.item(), self.rewards_mean.item())
        self.assertEqual(reward_raw_std.item(), self.rewards_std.item())

        # Update normalisation metrics with the batch
        self.learner._normalise_rewards()

        # extract normalised rewards for further verification
        reward_mean = self.learner.rollout.transitions['rewards'].mean()
        reward_std = self.learner.rollout.transitions['rewards'].std().clamp(min=1e-8)

        # Result printout for verification
        print(f"\n{'Metric':<20} {'Calculated':<15} {'Target':<15} {'Difference':<15}")
        print("-" * 65)
        print(f"{'Raw GAE Mean':<20} {reward_raw_mean:<15.5f} {self.rewards_mean.item():<15.5f} {abs(reward_raw_mean - self.rewards_mean.item()):<15.5f}")
        print(f"{'Raw GAE Std':<20} {reward_raw_std:<15.5f} {self.rewards_std.item():<15.5f} {abs(reward_raw_std - self.rewards_std.item()):<15.5f}")
        print(f"{'GAE Mean':<20} {reward_mean:<15.5f} {0.0:<15.5f} {abs(reward_mean - 0.0):<15.5f}")
        print(f"{'GAE Std':<20} {reward_std:<15.5f} {1.0:<15.5f} {abs(reward_std - 1.0):<15.5f}\n")

         # extract normalised rewards for further verification
        print(f"{'Normalised Rewards':<20} {self.learner.rollout.transitions['rewards']}")
        print(f"{'Expected Normalised':<20} {self.normalised_rewards}\n")

        # Assertions
        self.assertAlmostEqual(self.learner.rollout.transitions['rewards'].mean(), self.normalised_rewards.mean().item(), places=3)
        self.assertAlmostEqual(reward_raw_std.item(), self.rewards_std.item(), places=3)
        self.assertAlmostEqual(reward_mean.item(), 0.0, places=3)
        self.assertAlmostEqual(reward_std.item(), 1.0, places=3)
