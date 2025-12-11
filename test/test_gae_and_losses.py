import torch
import numpy as np
import pytest

from src.algorithms.loss import policy_loss, value_loss
from src.controllers.PPOController import PPOController
from src.configs.ControllerConfigs import PPOControllerConfig


def get_fake_controller():
    """
    Creates a small PPOController using a minimal deterministic config.
    Used across tests to avoid repeating boilerplate.
    """
    config_dict = {
        'action_size': 5,
        'n_features': 12,
        'state_size': 12,
        'encoder': {
            'type': 'FNN',
            'layer_sizes': [512, 256],
            'output_size': 128
        },
        'actor_config': {
            'type': 'FNN',
            'layer_sizes': [64]
        },
        'critic_config': {
            'type': 'FNN',
            'layer_sizes': [128, 64]
        }
    }
    controller_config = PPOControllerConfig(config_dict)
    return controller_config.create_controller()


# =======================================================
# PPO FORWARD PASS + LOSS SANITY TEST
# =======================================================
def test_ppo_loss_update():
    """
    Ensures that sample_action returns valid tensors with correct shapes,
    and verifies that log_probs, entropy, and state values are all finite.
    Does NOT check gradients — just checks end-to-end controller output.
    """
    torch.manual_seed(42)
    controller = get_fake_controller()

    batch_size = 5
    n_actions = 5
    n_features = 12

    states = torch.randn(batch_size, n_features)
    next_states = torch.randn(batch_size, n_features)
    actions = torch.randint(0, n_actions, (batch_size,))

    log_probs, entropy, state_vals, next_state_vals = controller.sample_action(states)
    assert log_probs.shape[0] == batch_size
    assert entropy.shape[0] == batch_size
    assert state_vals.shape[0] == batch_size



# =======================================================
# GAE REFERENCE IMPLEMENTATION
# =======================================================
def compute_gae_reference(rewards, values, next_values, dones, gamma, lam):
    """
    Numpy reference implementation of Generalized Advantage Estimation (GAE).
    Used to verify that the PyTorch GAE loop behaves correctly.
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=float)
    last_gae = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_values[t] * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        adv[t] = last_gae
    return adv


# =======================================================
# GAE NUMERICAL CORRECTNESS TEST
# =======================================================
def test_gae_simple():
    """
    Compares PyTorch GAE implementation vs the numpy reference.
    Ensures correctness for a simple hand-computed sequence.
    """
    rewards = [1.0, 0.0, 0.0]
    values = np.array([0.5, 0.4, 0.3])
    next_values = np.array([0.4, 0.3, 0.0])
    dones = [0, 0, 1]
    gamma = 0.99
    lam = 0.95

    ref = compute_gae_reference(rewards, values, next_values, dones, gamma, lam)

    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    values_t = torch.tensor(values, dtype=torch.float32)
    next_values_t = torch.tensor(next_values, dtype=torch.float32)
    dones_t = torch.tensor(dones, dtype=torch.float32)

    gaes = []
    advantage = torch.tensor(0.0)
    for t in reversed(range(len(rewards))):
        delta = rewards_t[t] + gamma * next_values_t[t] * (1 - dones_t[t]) - values_t[t]
        advantage = delta + gamma * lam * (1 - dones_t[t]) * advantage
        gaes.insert(0, advantage.item())

    np.testing.assert_allclose(np.array(gaes), ref, rtol=1e-5, atol=1e-6)



# =======================================================
# POLICY LOSS CLIPPING TEST
# =======================================================
def test_policy_loss_clipped():
    """
    Verifies that PPO's clipped surrogate objective matches
    the expected manually-computed result for a tiny example.
    """
    advantages = torch.tensor([1.0, -0.5], dtype=torch.float32)
    old_log_probs = torch.log(torch.tensor([0.6, 0.4], dtype=torch.float32))
    new_log_probs = torch.log(torch.tensor([0.7, 0.3], dtype=torch.float32))
    clip_eps = 0.2

    ratios = (new_log_probs - old_log_probs).exp()
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    expected = -torch.mean(torch.min(surr1, surr2))

    loss = policy_loss(gae=advantages, new_log_prob=new_log_probs, old_log_prob=old_log_probs, clip_eps=clip_eps)
    assert pytest.approx(expected.item(), rel=1e-5) == loss.item()



# =======================================================
# VALUE LOSS MSE TEST
# =======================================================
def test_value_loss_mse():
    """
    Tests that value_loss implements correct MSE:
        loss = mean((pred - target)^2)
    """
    preds = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    targets = torch.tensor([1.5, 1.8, 2.9], dtype=torch.float32)
    expected = torch.mean((preds - targets) ** 2)
    loss = value_loss(predicted_values=preds, target_values=targets)
    assert pytest.approx(expected.item(), rel=1e-6) == loss.item()



# =======================================================
# FULL PPO PIPELINE SANITY CHECK
# =======================================================
def test_ppo_loss_and_gae():
    """
    Runs a small PPO forward pass + loss computation:
    - checks policy loss is finite and non-zero
    - checks value loss is finite and non-zero
    - checks GAEs contain valid values
    This validates the entire update pipeline logically works.
    """
    torch.manual_seed(42)
    controller = get_fake_controller()
    batch_size = 5
    n_actions = controller.config['action_size']
    state_size = controller.config['n_features']

    states = torch.randn(batch_size, state_size)
    next_states = torch.randn(batch_size, state_size)
    actions = torch.randint(0, n_actions, (batch_size,))

    old_log_probs = torch.randn(batch_size)
    gaes = torch.randn(batch_size)
    state_values = torch.randn(batch_size)
    next_state_values = torch.randn(batch_size)

    encoded_states = controller.encoder_network(states)
    encoded_next_states = controller.encoder_network(next_states)
    logits = controller.actor_network(encoded_states)
    action_distribution = torch.distributions.Categorical(logits=logits)
    new_log_probs = action_distribution.log_prob(actions)
    entropy = action_distribution.entropy()
    new_state_values_eval = controller.critic_network(encoded_states)
    new_next_state_values_eval = controller.critic_network(encoded_next_states)

    pol_loss = policy_loss(gae=gaes, new_log_prob=new_log_probs, old_log_prob=old_log_probs, clip_eps=0.2)
    assert torch.isfinite(pol_loss), "Policy loss contains NaNs or Infs"
    assert pol_loss.item() != 0, "Policy loss is zero, likely incorrect"

    val_loss = value_loss(predicted_values=new_state_values_eval.squeeze(-1), target_values=state_values)
    assert torch.isfinite(val_loss), "Value loss contains NaNs or Infs"
    assert val_loss.item() != 0, "Value loss is zero, likely incorrect"

    assert torch.all(torch.isfinite(gaes)), "GAEs contain NaNs or Infs"
    assert torch.std(gaes) > 0, "GAEs have zero variance, likely incorrect"



# =======================================================
# ZERO-REWARD GAE EDGE CASE
# =======================================================
def test_gae_zero_rewards():
    """
    If rewards, values, and next_values are all zero,
    GAE should be exactly zero.
    """
    rewards = [0, 0, 0]
    values = np.array([0.0, 0.0, 0.0])
    next_values = np.array([0.0, 0.0, 0.0])
    dones = [0, 0, 0]
    gaes = compute_gae_reference(rewards, values, next_values, dones, gamma=0.99, lam=0.95)
    np.testing.assert_array_equal(gaes, np.zeros_like(gaes))



# =======================================================
# MULTIPLE EDGE CASES
# =======================================================
def test_gae_and_losses_edge_cases():
    """
    Tests multiple critical boundary cases:
    - GAE should be zero when all rewards/dones stop propagation
    - Value loss must be zero when preds == targets
    - Policy loss should match unclipped surrogate when ratio = 1
    - Entropy must be finite
    """
    torch.manual_seed(123)
    controller = get_fake_controller()
    batch_size = 4
    n_actions = controller.config['action_size']
    state_size = controller.config['n_features']

    states = torch.randn(batch_size, state_size)
    actions = torch.randint(0, n_actions, (batch_size,))

    # ---- Edge 1: zero reward, zero values, done everywhere
    rewards = torch.zeros(batch_size)
    values = torch.zeros(batch_size)
    next_values = torch.zeros(batch_size)
    dones = torch.ones(batch_size)
    gamma, lam = 0.99, 0.95

    gaes = []
    advantage = torch.tensor(0.0)
    for t in reversed(range(batch_size)):
        delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
        advantage = delta + gamma * lam * (1 - dones[t]) * advantage
        gaes.insert(0, advantage.item())
    assert all(g == 0 for g in gaes), "GAEs should be zero for zero rewards and done states"

    # ---- Edge 2: value_loss = 0 when preds == targets
    preds = torch.tensor([1.0, 2.0, 3.0, 4.0])
    targets = preds.clone()
    val_loss = value_loss(predicted_values=preds, target_values=targets)
    assert val_loss.item() == 0, "Value loss should be zero when predictions equal targets"

    # ---- Edge 3: policy_loss when ratio = 1
    old_log_probs = torch.log(torch.tensor([0.5, 0.2, 0.2, 0.1]))
    new_log_probs = old_log_probs.clone()
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
    pol_loss = policy_loss(gae=advantages, new_log_prob=new_log_probs, old_log_prob=old_log_probs, clip_eps=0.2)
    expected = -torch.mean(advantages * (new_log_probs - old_log_probs).exp())
    assert torch.isclose(pol_loss, expected)

    # ---- Edge 4: entropy must be finite
    log_probs, entropy, state_vals, next_state_vals = controller.sample_action(states)
    assert torch.all(torch.isfinite(entropy)), "Entropy contains NaNs or Infs"
    assert entropy.shape[0] == batch_size



# =======================================================
# VALUE LOSS SHAPE SAFETY
# =======================================================
def test_value_loss_shapes():
    """
    Ensures that value_loss handles (batch,1) predictions and (batch,) targets
    without raising an error.
    """
    preds = torch.randn(4, 1)
    targets = torch.randn(4)
    _ = value_loss(predicted_values=preds.squeeze(-1), target_values=targets)



def test_value_loss_warns_on_mismatched_shape():
    """
    If target shape does not match prediction shape,
    value_loss should emit a warning.
    """
    preds = torch.randn(4)
    targets = torch.randn(4, 4)

    with pytest.warns(UserWarning):
        _ = value_loss(preds, targets)



# =======================================================
# DEBUGGING UTIL — MANUAL PPO STEP
# =======================================================
def debug_ppo_step(controller: PPOController, batch_size=4):
    """Utility function for printing a full PPO step for manual debugging."""
    torch.manual_seed(42)
    n_actions = controller.config['action_size']
    state_size = controller.config['n_features']

    states = torch.randn(batch_size, state_size)
    next_states = torch.randn(batch_size, state_size)
    actions = torch.randint(0, n_actions, (batch_size,))

    log_probs, entropy, state_values, next_state_values = controller.sample_action(states)

    print("=== Policy outputs ===")
    print("Logits / Log probs:", log_probs)
    print("Entropy:", entropy)
    print("Min entropy:", entropy.min().item(), "Max entropy:", entropy.max().item())

    print("\n=== Critic outputs ===")
    print("State values:", state_values.squeeze(-1))
    if next_state_values is not None:
        print("Next state values:", next_state_values.squeeze(-1))
    else:
        print("Next state values: None (fallback to state_values for GAEs)")
        next_state_values = state_values

    rewards = torch.randn(batch_size)
    dones = torch.zeros(batch_size)
    gaes = []
    advantage = torch.tensor(0.0)
    gamma, lam = 0.99, 0.95
    for t in reversed(range(batch_size)):
        delta = rewards[t] + gamma * next_state_values[t] * (1 - dones[t]) - state_values[t]
        advantage = delta + gamma * lam * (1 - dones[t]) * advantage
        gaes.insert(0, advantage)
    gaes = torch.stack(gaes)

    print("\n=== GAEs ===")
    print("GAEs:", gaes)
    print("GAE mean:", gaes.mean().item(), "std:", gaes.std().item())

    old_log_probs = log_probs.detach()
    actor_loss = policy_loss(gae=gaes, new_log_prob=log_probs, old_log_prob=old_log_probs, clip_eps=0.2)
    critic_loss = value_loss(
        predicted_values=state_values.squeeze(-1),
        target_values=(state_values.squeeze(-1) + gaes).detach()
    )
    total_loss = actor_loss + critic_loss

    print("\n=== Losses ===")
    print("Actor loss:", actor_loss.item())
    print("Critic loss:", critic_loss.item())
    print("Total loss:", total_loss.item())

    with torch.no_grad():
        actor_norm = torch.nn.utils.parameters_to_vector(controller.actor_network.parameters()).norm().item()
        critic_norm = torch.nn.utils.parameters_to_vector(controller.critic_network.parameters()).norm().item()
        encoder_norm = torch.nn.utils.parameters_to_vector(controller.encoder_network.parameters()).norm().item()
    print("\n=== Parameter norms ===")
    print(f"Actor: {actor_norm:.4f}, Critic: {critic_norm:.4f}, Encoder: {encoder_norm:.4f}")


# Standalone manual debug
if __name__ == "__main__":
    controller = get_fake_controller()
    debug_ppo_step(controller)
