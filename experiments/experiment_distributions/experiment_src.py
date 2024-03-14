import os, sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(root_dir)

import numpy as np
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

from rl_envs_forge.envs.grid_world.grid_world import GridWorld
from rl_envs_forge.envs.grid_world.grid_world import Action

from overfitting.src.utils import create_random_policy
from overfitting.src.policy_iteration import random_policy_evaluation_q_stochastic

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import logging


def make_env(rows, cols, start_state, p_success, terminal_states, seed):
    return GridWorld(
        rows=rows,
        cols=cols,
        start_state=start_state,
        walls=None,
        p_success=p_success,
        terminal_states=terminal_states,
        seed=seed,
        rewards={
            "valid_move": 0,
            "wall_collision": 0,
            "out_of_bounds": 0,
            "default": 0.0,
        },
    )


def softmax(logits, tau, lower_bound):
    logits = np.array(
        logits
    )  # Ensure logits is a NumPy array for consistent operations
    logits -= np.max(logits)  # Improves numerical stability
    exp_logits = np.exp(logits / tau)
    softmax_probs = exp_logits / np.sum(exp_logits)
    adjusted_probs = np.clip(
        softmax_probs + lower_bound, 0, 1
    )  # Ensure probabilities are valid
    adjusted_probs /= np.sum(adjusted_probs)
    return adjusted_probs


def generate_transitions_observations(transitions_list, num_steps, tau, lower_bound=0):
    dset_size = len(transitions_list)
    logits = np.random.uniform(0, 1, size=dset_size)
    prob_dist = softmax(logits, tau, lower_bound)

    sampled_indices = np.random.choice(
        len(transitions_list), size=num_steps, p=prob_dist
    )
    sampled_transitions = [transitions_list[i] for i in sampled_indices]
    return sampled_transitions


def td_learning(states, actions, training_data, alpha, gamma, epsilon):
    # Initialize Q as a nested dictionary for all state-action pairs
    Q = {state: {action: 0 for action in actions} for state in states}

    # TD Learning process
    converged = False
    iterations = 0
    max_iterations = 10000  # Prevent infinite loops

    while not converged and iterations < max_iterations:
        max_delta = 0
        for current_state, action, next_state, reward, done, _ in training_data:
            current_state_dict = tuple(
                current_state
            )  # Ensure the state is hashable and matches the dict keys

            # Skip the update if the current state is terminal or not in the Q dictionary
            if current_state_dict not in Q:
                continue

            if done:
                # No further action is taken if the next state is terminal; assume Q-value is zero for terminal states
                td_target = reward
            else:
                next_state_dict = tuple(next_state)
                # Ensure next_state is in Q; if not, treat it as terminal with zero future rewards
                if next_state_dict in Q:
                    td_target = reward + gamma * max(Q[next_state_dict].values())
                else:
                    td_target = reward

            # TD Update for non-terminal states
            old_q = Q[current_state_dict][action]
            Q[current_state_dict][action] += alpha * (td_target - old_q)

            # Track the maximum change in Q-values for convergence check
            delta = abs(old_q - Q[current_state_dict][action])
            max_delta = max(max_delta, delta)

        if max_delta < epsilon:
            converged = True

        iterations += 1

    return Q


def compute_regret(Q_star, Q_td, states, actions):
    total_regret = 0
    regrets = {state: {action: 0 for action in actions} for state in states}

    for state in states:
        for action in actions:
            # Calculate regret for the current state-action pair
            regret = Q_star[state].get(action, 0) - Q_td[state].get(action, 0)
            regrets[state][action] = regret

            # Update the total regret
            total_regret += abs(regret)  # Using absolute value of regret

    return total_regret, regrets


class TransitionDataset(Dataset):
    def __init__(self, transitions):
        self.transitions = transitions

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        state, action, next_state, reward, done, _ = self.transitions[idx]
        return (
            torch.tensor(state, dtype=torch.float),
            action,
            torch.tensor(next_state, dtype=torch.float),
            reward,
            done,
        )


def train_dqn(
    dqn,
    transitions,
    Q_pi_random,
    states,
    actions,
    gamma,
    epsilon,
    batch_size,
    max_iterations,
    logger=None,
):
    if logger is None:
        logger = logging.getLogger(__name__)

    dataset = TransitionDataset(transitions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(dqn.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    loss_record = []

    for epoch in range(max_iterations):
        total_loss = 0
        abs_diffs = (
            []
        )  # List to store absolute differences for the expected value calculation

        for state, action, next_state, reward, done in dataloader:
            optimizer.zero_grad()
            q_values = dqn(state)
            next_q_values = dqn(next_state)
            max_next_q_values = next_q_values.max(1)[0].detach()
            target_q_values = q_values.clone()

            for i in range(len(done)):
                update = (
                    reward[i] if done[i] else reward[i] + gamma * max_next_q_values[i]
                )
                target_q_values[i, action[i]] = update

            loss = loss_fn(q_values, target_q_values)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # After updating the network, compute the expected value of |Q_train - Q_pi_random| over the validation set
        for state in states:
            state_tensor = torch.tensor([state], dtype=torch.float).unsqueeze(
                0
            )  # Assume state is a list or tuple
            q_values = dqn(state_tensor).detach().numpy().squeeze()
            for action in actions:
                abs_diff = abs(q_values[action] - Q_pi_random[state][action])
                abs_diffs.append(abs_diff)

        expected_value = np.mean(abs_diffs)  # Compute the mean of absolute differences

        loss_record.append((epoch, total_loss, expected_value))

        # if expected_value < epsilon:
        #     print(
        #         f"Converged after {epoch + 1} epochs with expected value {expected_value}"
        #     )
        #     logger.info(f"Converged after {epoch + 1} epochs with expected value {expected_value}")
        #     break

    logger.info(
        f"Exiting after {epoch + 1} epochs with expected value {expected_value}"
    )

    return loss_record


def compute_bellmans_error(dqn, validation_transitions, gamma=0.99):
    dqn.eval()  # Set the DQN to evaluation mode
    bellmans_errors = []

    with torch.no_grad():  # No need to compute gradients
        for state, action, next_state, reward, done, _ in validation_transitions:
            state_tensor = torch.tensor([state], dtype=torch.float)
            next_state_tensor = torch.tensor([next_state], dtype=torch.float)

            # Predicted Q-value for the current state and action
            q_value = dqn(state_tensor)[0, action]

            # Target Q-value using Bellman equation
            if done:
                target_q_value = reward
            else:
                next_q_values = dqn(next_state_tensor)
                max_next_q_value = torch.max(next_q_values)
                target_q_value = reward + gamma * max_next_q_value

            # Bellman's error
            bellman_error = abs(q_value - target_q_value)
            bellmans_errors.append(bellman_error.item())

    # Compute the mean Bellman's error over all transitions
    mean_bellmans_error = sum(bellmans_errors) / len(bellmans_errors)
    return mean_bellmans_error


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        return self.network(x)


def on_policy_loss(predictions, targets):
    # Implement the on-policy error calculation here
    # This is a placeholder for the actual loss calculation
    loss = torch.nn.functional.mse_loss(predictions, targets)
    return loss


def run_sampling_regret_experiment(
    tau,
    seed,
    rows,
    cols,
    start_state,
    p_success,
    terminal_states,
    num_steps,
    epsilon,
    gamma,
    lower_bound_softmax,
    batch_size,
    train_max_iterations,
    logger=None,
):
    np.random.seed(seed)

    if logger is None:
        logger = logging.getLogger(__name__)

    env = make_env(rows, cols, start_state, p_success, terminal_states, seed)

    states = list(set([s for s, _ in env.mdp.keys()]))
    actions = list(set([a for _, a in env.mdp.keys()]))
    random_policy = create_random_policy(states, actions)

    Q = {state: {action: 0 for action in actions} for state in states}
    Q_pi_random = random_policy_evaluation_q_stochastic(
        states, actions, random_policy, Q, env.mdp, gamma, epsilon
    )

    transitions_list = [(key[0], key[1], *value[0]) for key, value in env.mdp.items()]
    transitions_train, transitions_val = train_test_split(
        transitions_list, test_size=0.2, random_state=seed
    )
    sampled_transitions_train = generate_transitions_observations(
        transitions_train,
        num_steps,
        tau=tau,
        lower_bound=lower_bound_softmax / len(transitions_train),
    )

    ### Training
    input_size = len(states[0])  # Or another way to represent the size of your input
    output_size = len(actions)

    # Initialize the DQN
    dqn = DQN(input_size, output_size)

    loss_record = train_dqn(
        dqn,
        sampled_transitions_train,
        Q_pi_random,
        states,
        actions,
        gamma,
        epsilon,
        batch_size=batch_size,
        max_iterations=train_max_iterations,
        logger=logger,
    )

    bm_error = compute_bellmans_error(
        dqn, validation_transitions=transitions_val, gamma=gamma
    )

    return loss_record, bm_error


def generate_random_policy_transitions(
    transitions_train, num_steps, env, actions, seed, logger
):
    np.random.seed(seed)
    transitions = []

    # Convert transitions_train to a set for efficient lookup
    transitions_train_set = {
        (s, a, ns, r, d): None for s, a, ns, r, d, _ in transitions_train
    }

    while len(transitions) < num_steps:
        state = env.reset()
        logger.info(f"State: {state}, reset environment")
        done = False
        while not done:
            action = np.random.choice(actions)
            next_state, reward, done, _, _ = env.step(action)
            transition = (state, action, next_state, reward, done)

            # Only add if the transition is in the transitions_train_set
            if transition[:5] in transitions_train_set:
                transitions.append(transition + (1,))  # adding probability

            state = next_state
            if len(transitions) >= num_steps:
                break

    return transitions


def train_dqn_with_policy_evaluation(
    dqn,
    transitions,
    states,
    actions,
    gamma,
    epsilon,
    batch_size,
    max_iterations,
    logger=None,
):
    if logger is None:
        logger = logging.getLogger(__name__)

    dataset = TransitionDataset(transitions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(dqn.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    loss_record = []

    for epoch in range(max_iterations):
        total_loss = 0
        for state, action, next_state, reward, done in dataloader:
            optimizer.zero_grad()
            q_values = dqn(state)

            next_q_values = dqn(next_state)
            # Compute expected Q-value for the next state using uniform random policy
            expected_next_q_values = next_q_values.mean(dim=1)

            target_q_values = q_values.clone()
            for i in range(len(done)):
                update = (
                    reward[i]
                    if done[i]
                    else reward[i] + gamma * expected_next_q_values[i]
                )
                target_q_values[i, action[i]] = update

            loss = loss_fn(q_values, target_q_values)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        loss_record.append((epoch, total_loss, 0))

        logger.info(f"Epoch {epoch + 1}, Total Loss: {total_loss}")

    return loss_record


def run_sampling_regret_experiment_with_policy_evaluation(
    tau,
    seed,
    rows,
    cols,
    start_state,
    p_success,
    terminal_states,
    num_steps,
    epsilon,
    gamma,
    lower_bound_softmax,
    batch_size,
    train_max_iterations,
    logger=None,
):
    np.random.seed(seed)

    if logger is None:
        logger = logging.getLogger(__name__)

    env = make_env(rows, cols, start_state, p_success, terminal_states, seed)

    states = list(set([s for s, _ in env.mdp.keys()]))
    actions = list(set([a for _, a in env.mdp.keys()]))

    transitions_list = [(key[0], key[1], *value[0]) for key, value in env.mdp.items()]
    transitions_train, transitions_val = train_test_split(
        transitions_list, test_size=0.2, random_state=seed
    )
    sampled_transitions_train = generate_transitions_observations(
        transitions_train,
        num_steps,
        tau=tau,
        lower_bound=lower_bound_softmax / len(transitions_train),
    )

    ### Training
    input_size = len(states[0])  # Or another way to represent the size of your input
    output_size = len(actions)

    # Initialize the DQN
    dqn_random_policy = DQN(input_size, output_size)

    loss_record_random_policy = train_dqn_with_policy_evaluation(
        dqn_random_policy,
        sampled_transitions_train,
        states,
        actions,
        gamma,
        epsilon,
        batch_size,
        train_max_iterations,
        logger,
    )

    bm_error = compute_bellmans_error(
        dqn_random_policy, validation_transitions=transitions_val, gamma=gamma
    )

    return loss_record_random_policy, bm_error


def run_baseline_random_policy_experiment(
    tau,
    seed,
    rows,
    cols,
    start_state,
    p_success,
    terminal_states,
    num_steps,
    epsilon,
    gamma,
    lower_bound_softmax,
    batch_size,
    train_max_iterations,
    logger=None,
):
    np.random.seed(seed)

    if logger is None:
        logger = logging.getLogger(__name__)

    env = make_env(rows, cols, start_state, p_success, terminal_states, seed)

    states = list(set([s for s, _ in env.mdp.keys()]))
    actions = list(set([a for _, a in env.mdp.keys()]))

    transitions_list = [(key[0], key[1], *value[0]) for key, value in env.mdp.items()]
    transitions_train, transitions_val = train_test_split(
        transitions_list, test_size=0.2, random_state=seed
    )

    random_policy_transitions = generate_random_policy_transitions(
        transitions_train, num_steps, env, actions, seed, logger
    )

    ### Training
    input_size = len(states[0])  # Or another way to represent the size of your input
    output_size = len(actions)

    # Initialize the DQN
    dqn_random_policy = DQN(input_size, output_size)

    loss_record_random_policy = train_dqn_with_policy_evaluation(
        dqn_random_policy,
        random_policy_transitions,
        states,
        actions,
        gamma,
        epsilon,
        batch_size,
        train_max_iterations,
        logger,
    )

    bm_error = compute_bellmans_error(
        dqn_random_policy, validation_transitions=transitions_val, gamma=gamma
    )

    return loss_record_random_policy, bm_error
