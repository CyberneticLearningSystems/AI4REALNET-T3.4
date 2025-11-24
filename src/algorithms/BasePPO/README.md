# Base PPO
This module contains all the files necessary to run a very simply PPO algorithm, with controller definition, environment configuration and learner configuration

## Controller Configuration
Controller Structure: 
1. **Encoder Network:** performs latent space encoding, which is then passed to the actor and critic heads of the network
2. **Actor Head:** decodes latent space to action logits, which can then be sampled (randomly from the distribution) or selected (choose highest probability action)
3. **Critic Head:** decodes latent space to state value



## RolloutBuffer
A very simple version of the ``MultiAgentRolloutBuffer`` has been programmed to gather the following variables for each transition: 
1. ``state`` $\rightarrow$ the initial state of the environment
2. ``action`` $\rightarrow$ the action chosen in this state
3. ``log_prob`` $\rightarrow$ the log probability of the action chosen (just the logarithm of the probability as a value between 0 and 1)
4. ``next_state`` $\rightarrow$ the state of the environment after executing the action(s)
5. ``reward`` $\rightarrow$ the reward achieved by choosing the action in the initial state
6. ``done`` $\rightarrow$ a boolean indicating if the agent finished it's rollout after this action (helps with bootstrapping)

To calculate the generalised advantage estimators (GAEs), trajectories must be mappable to agents, meaning that the rollouts have the size ``(episode_length x n_agents)``. They can only be squeezed to ``(episode_length * n_agents)`` once GAEs have been calculated. Once GAEs are calculated, the rollout can be stacked and returned as a single vector. 

