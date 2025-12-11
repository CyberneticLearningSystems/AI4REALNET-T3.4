import torch
from torch import nn
from src.controllers.PPOController import PPOController
from src.algorithms.loss import policy_loss, value_loss

    def test_ppo_loss_update():
        torch.manual_seed(42)

        # --- Setup a small fake environment ---
        n_agents = 2
        state_size = 4
        n_actions = 3
        batch_size = 5

        # Fake states and next states
        states = torch.randn(batch_size, state_size)
        next_states = torch.randn(batch_size, state_size)

        # Fake actions, rewards, log_probs, values
        actions = torch.randint(0, n_actions, (batch_size,))
        old_log_probs = torch.randn(batch_size)
        gaes = torch.randn(batch_size)
        state_values = torch.randn(batch_size)
        next_state_values = torch.randn(batch_size)
        rewards = torch.randn(batch_size)
        dones = torch.zeros(batch_size)

        # --- Setup controller and optimizer ---
        controller = PPOController(state_size=state_size, action_size=n_actions)
        optimizer = torch.optim.Adam(controller.get_parameters(), lr=1e-3)

        # Forward pass
        logits = controller.actor_network(states)
        action_dist = torch.distributions.Categorical(logits=logits)
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        values = controller.critic_network(states).squeeze(-1)
        next_values = controller.critic_network(next_states).squeeze(-1)

        # --- Compute losses ---
        actor_loss = policy_loss(gae=gaes, new_log_prob=log_probs, old_log_prob=old_log_probs, clip_eps=0.2)
        value_targets = gaes + state_values
        critic_loss = value_loss(predicted_values=values, target_values=value_targets)
        total_loss = actor_loss + critic_loss

        # --- Backprop ---
        optimizer.zero_grad()
        total_loss.backward()

        # Check that gradients exist
        actor_grads = [p.grad for p in controller.actor_network.parameters()]
        critic_grads = [p.grad for p in controller.critic_network.parameters()]
        assert all(g is not None for g in actor_grads), "Actor gradients are missing"
        assert all(g is not None for g in critic_grads), "Critic gradients are missing"

        # --- Optimizer step ---
        before_params = [p.clone() for p in controller.actor_network.parameters()]
        optimizer.step()
        after_params = [p for p in controller.actor_network.parameters()]

        # Check that parameters changed
        param_deltas = [torch.sum(torch.abs(a - b)) for a, b in zip(before_params, after_params)]
        assert any(delta.item() > 0 for delta in param_deltas), "Actor parameters did not update"

        print("PPO loss and optimizer test passed.")

    def test_policy_forward_pass(env_config):
        """
        Test whether the PPOController can perform a forward pass
        using environment observation & action sizes.
        """

        env = env_config.create_env()
        obs, _ = env.reset()
        state_size = obs.shape[0]
        action_size = env.action_space[0].n  # multi-agent → pick agent 0

        # ---- BUILD VALID PPO CONFIG ---- #
        config = {
            "state_size": state_size,
            "action_size": action_size,

            "encoder": {
                "hidden_layers": [64],
                "output_size": 32,
                "activation": "relu",
            },

            "actor_config": {
                "hidden_layers": [64],
                "activation": "relu",
            },

            "critic_config": {
                "hidden_layers": [64],
                "activation": "relu",
            },
        }

        controller = PPOController(config)

        # Convert obs to tensor
        state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        # Run forward pass
        action, log_prob, value, extras = controller.sample_action(state_tensor)

        assert action.shape == (1,)
        assert log_prob.shape == (1,)
        assert value.shape == (1, 1)
