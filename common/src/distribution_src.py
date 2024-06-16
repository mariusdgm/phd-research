import os, sys

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(root_dir)

import numpy as np
from sklearn.model_selection import train_test_split
import random
import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

from rl_envs_forge.envs.grid_world.grid_world import GridWorld
from rl_envs_forge.envs.grid_world.grid_world import Action

from common.src.utils import create_random_policy
from common.src.policy_iteration import random_policy_evaluation_q_stochastic
from common.src.experiment_utils import seed_everything
from common.src.dqn.replay_buffer import ReplayBuffer
from common.src.simple_dqn_agent import AgentDQN
from common.src.frequency_normalization import (
    get_frequency_scaling,
    normalize_frequencies,
)
from common.src.models import QNET


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LinearLR

import networkx as nx

import logging


def make_env(
    rows,
    cols,
    start_state,
    p_success,
    terminal_states,
    seed,
    walls=None,
    episode_length_limit=None,
):
    return GridWorld(
        rows=rows,
        cols=cols,
        start_state=start_state,
        walls=walls,
        p_success=p_success,
        terminal_states=terminal_states,
        seed=seed,
        rewards={
            "valid_move": 0,
            "wall_collision": 0,
            "out_of_bounds": 0,
            "default": 0.0,
        },
        episode_length_limit=episode_length_limit,
    )


def randomize_walls_positions(
    rows, columns, starting_cell, terminal_cells, ratio, seed
):
    """Randomly select cells where to place walls in the gridworld.
    Walls can't be placed in the starting cell or terminal cells.
    The number of wall cells is determined by the `percentage` parameter.

    Args:
        rows (int): Number of rows in the grid.
        columns (int): Number of columns in the grid.
        starting_cell (tuple): Coordinates of the starting cell (row, column).
        terminal_cells (list of tuples): Coordinates of the terminal cells.
        ratio (float): Ratio of total cells to be converted to walls.
        seed (int, optional): Seed for the random number generator.
    Returns:
        set: Set of tuples where each tuple represents coordinates (row, column) of a wall cell.
    """
    if seed is not None:
        random.seed(seed)

    total_cells = rows * columns
    wall_count = int(ratio * total_cells)

    # Create a set of all possible cell coordinates
    all_cells = {(r, c) for r in range(rows) for c in range(columns)}

    # Remove the starting and terminal cells from possible wall locations
    forbidden_cells = set(terminal_cells)
    forbidden_cells.add(starting_cell)
    available_cells = list(all_cells - forbidden_cells)

    # Adjust the wall_count if necessary
    if wall_count > len(available_cells):
        wall_count = len(available_cells)

    # Randomly select wall_count cells to be walls
    wall_cells = set(random.sample(available_cells, wall_count))

    return wall_cells


def softmax(logits, tau):
    logits = np.array(logits)  # Ensure logits is a NumPy array for consistency
    logits -= np.max(logits)  # Improves numerical stability
    exp_logits = np.exp(logits / tau)
    softmax_probs = exp_logits / np.sum(exp_logits)
    return softmax_probs


def generate_transitions_observations(
    transitions_list, num_steps, tau, min_samples=None
):
    dset_size = len(transitions_list)

    # Validate min_samples if provided
    if min_samples is not None:
        if not isinstance(min_samples, int):
            raise ValueError("min_samples must be an integer")
        if min_samples * dset_size > num_steps:
            raise ValueError(
                "min_samples times length of transitions exceeds num_steps"
            )

        # Directly select each element in transitions_list min_samples times, considering their structure
        repeated_transitions = [
            transition for transition in transitions_list for _ in range(min_samples)
        ]
        remaining_steps = num_steps - (min_samples * dset_size)
    else:
        repeated_transitions = []
        remaining_steps = num_steps

    sampled_transitions = (
        repeated_transitions  # Start with the manually repeated transitions
    )

    if remaining_steps > 0:
        # Prepare logits for the softmax sampling; logits could be based on some criteria or just random
        logits = np.random.uniform(0, 1, size=dset_size)
        prob_dist = softmax(logits, tau)

        # Sample the remaining transitions based on the softmax distribution
        sampled_indices = np.random.choice(
            dset_size, size=remaining_steps, p=prob_dist, replace=True
        )
        sampled_transitions += [transitions_list[i] for i in sampled_indices]

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
    def __init__(self, transitions, inverse_frequency_scaling=None):
        self.transitions = transitions
        self.inverse_frequency_scaling = inverse_frequency_scaling

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):

        state, action, next_state, reward, done, _ = self.transitions[idx]
        state_tensor = torch.tensor(state, dtype=torch.float)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float)
        reward_tensor = torch.tensor(reward, dtype=torch.float)

        if self.inverse_frequency_scaling:
            transition_for_counting = (state, action, next_state, reward, int(done))
            scale_factor = self.inverse_frequency_scaling.get(
                transition_for_counting, 1.0
            )
        else:
            scale_factor = 1.0  # Default scale factor when no scaling is provided

        return (
            state_tensor,
            action,
            next_state_tensor,
            reward_tensor,
            done,
            scale_factor,
        )


def train_net_with_neural_fitting(
    net,
    transitions,
    gamma,
    batch_size,
    max_iterations,
    frequency_scaling=False,
    mode="max",
    logger=None,
):
    """
    Trains a neural network using the Neural Fitting algorithm.

    Args:
        net (torch.nn.Module): The neural network to be trained.
        transitions (list): A list of transition tuples (s, a, r, s').
        Q_pi_random (torch.Tensor): A tensor containing random Q-values for the given policy.
        states (list): A list of states.
        actions (list): A list of actions.
        gamma (float): The discount factor.
        epsilon (float): The exploration rate.
        batch_size (int): The batch size for training.
        max_iterations (int): The maximum number of training iterations.
        frequency_scaling (bool, optional): Whether to use frequency scaling. Defaults to False.
        mode (str, optional): The mode of the Bellman equation. Defaults to "max". Can be "max" or "mean".
        logger (logging.Logger, optional): The logger for logging training information. Defaults to None.

    Returns:
        list: A list of tuples containing the epoch number, total loss, and current learning rate.
    """

    if logger is None:
        logger = logging.getLogger(__name__)

    net.train()

    if frequency_scaling:
        inverse_frequency_scaling = get_frequency_scaling(transitions)
        dataset = TransitionDataset(transitions, inverse_frequency_scaling)
    else:
        dataset = TransitionDataset(transitions)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    # scheduler = LinearLR(
    #     optimizer, start_factor=1, end_factor=0, total_iters=max_iterations
    # )

    loss_record = []

    for epoch in range(max_iterations):
        total_loss = 0
        for state, action, next_state, reward, done, scale_factor in dataloader:
            optimizer.zero_grad()

            bellmans_errors = bellman_error(
                net, state, action, next_state, reward, done, mode=mode, gamma=gamma
            )
            net.train()

            if frequency_scaling:
                scale_factor = inverse_frequency_scaling.get(
                    (state, action, next_state, reward, done), 1
                )
                bellmans_errors = scale_factor * bellmans_errors

            loss = bellmans_errors.mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # scheduler.step()

        # current_lr = scheduler.get_last_lr()[0]
        current_lr = 0

        loss_record.append((epoch, total_loss, current_lr))

    return loss_record


def compute_validation_bellmans_error(
    model, validation_transitions, error_mode, gamma=0.99, logger=None
):
    if logger is None:
        logger = logging.getLogger(__name__)

    model.eval()  # Set the DQN to evaluation mode

    with torch.no_grad():
        # Prepare tensors for the whole batch
        states = torch.tensor([t[0] for t in validation_transitions], dtype=torch.float)
        actions = torch.tensor(
            [t[1] for t in validation_transitions], dtype=torch.int64
        )
        next_states = torch.tensor(
            [t[2] for t in validation_transitions], dtype=torch.float
        )
        rewards = torch.tensor(
            [t[3] for t in validation_transitions], dtype=torch.float
        )
        dones = torch.tensor([t[4] for t in validation_transitions], dtype=torch.bool)

        # Compute Bellman error
        bellmans_error = bellman_error(
            model,
            states,
            actions,
            next_states,
            rewards,
            dones,
            mode=error_mode,
            gamma=gamma,
        )

    result = bellmans_error.mean().item()
    if result == float("inf") or result == -float("inf") or np.isnan(result):
        logger.error(
            f"Invalid Bellman error: {result} Bellman error array {bellmans_error}. Data used: {validation_transitions}"
        )

    return result


def bellman_error(
    model, states, actions, next_states, rewards, dones, mode="max", gamma=0.99
):
    """
    Calculates the bellman error for a batch of transitions.

    This function is intended for use within a training loop, where the resulting
    error will be used for backpropagation.
    """

    model.eval()  # Ensure the model is in evaluation mode for this calculation

    # Predict Q-values for all states and next states
    q_values = model(states)
    next_q_values = model(next_states)

    # Compute max or mean Q-value for next states based on mode
    if mode == "max":
        next_q_values = next_q_values.detach().max(1)[0]
    elif mode == "mean":
        next_q_values = next_q_values.detach().mean(dim=1)

    # Compute target Q-values
    target_q_values = rewards + gamma * next_q_values * (~dones)

    # Select the Q-values for the actions taken
    action_q_values = q_values.gather(1, actions.long().unsqueeze(1)).squeeze(1)

    # Compute Bellman's error using MSE Loss
    loss_fn = nn.MSELoss(reduction="none")
    bellmans_error = loss_fn(action_q_values, target_q_values)

    return bellmans_error


def run_sampling_regret_experiment(
    tau,
    seed,
    run_id,
    rows,
    cols,
    start_state,
    p_success,
    terminal_states,
    num_steps,
    epsilon,
    gamma,
    min_samples,
    batch_size,
    train_max_iterations,
    neural_fit_mode,
    logger=None,
):
    if logger is None:
        logger = logging.getLogger(__name__)

    seed_everything(run_id)
    env = make_env(rows, cols, start_state, p_success, terminal_states, run_id)

    states = list(set([s for s, _ in env.mdp.keys()]))
    actions = list(set([a for _, a in env.mdp.keys()]))
    random_policy = create_random_policy(states, actions)
    Q = {state: {action: 0 for action in actions} for state in states}
    Q_pi_random = random_policy_evaluation_q_stochastic(
        states, actions, random_policy, Q, env.mdp, gamma, epsilon
    )

    transitions_list = [(key[0], key[1], *value[0]) for key, value in env.mdp.items()]
    transitions_train, transitions_val = generate_train_test_split_with_valid_path(
        transitions_list=transitions_list,
        start_state=start_state,
        terminal_states=terminal_states,
        seed=run_id,
    )
    sampled_transitions_train = generate_transitions_observations(
        transitions_train, num_steps, tau=tau, min_samples=min_samples
    )

    ### Training
    input_size = len(states[0])  # Or another way to represent the size of your input
    output_size = len(actions)
    seed_everything(seed)

    # Initialize the DQN
    qnet = QNET(input_size, output_size)
    loss_record = train_net_with_neural_fitting(
        qnet,
        sampled_transitions_train,
        Q_pi_random,
        states,
        actions,
        gamma,
        epsilon,
        batch_size=batch_size,
        max_iterations=train_max_iterations,
        frequency_scaling=False,
        mode=neural_fit_mode,
        logger=logger,
    )

    bm_error_validation = compute_validation_bellmans_error(
        qnet,
        validation_transitions=transitions_val,
        error_mode=neural_fit_mode,
        gamma=gamma,
        logger=logger,
    )

    bm_error_train = compute_validation_bellmans_error(
        qnet,
        validation_transitions=transitions_train,
        error_mode=neural_fit_mode,
        gamma=gamma,
        logger=logger,
    )

    return loss_record, bm_error_validation, bm_error_train


def generate_random_policy_transitions(transitions_list, num_steps, env, actions):
    transitions = []

    # Convert transitions_train to a set with float rewards for efficient lookup
    transitions_train_set = set(
        (s, a.value, ns, float(r), d) for s, a, ns, r, d, _ in transitions_list
    )
    [t for t in transitions_train_set if t[3] == 1]

    while len(transitions) < num_steps:
        state = env.reset()
        # logger.info(f"State: {state}, reset environment")
        done = False
        while not done:
            action = np.random.choice(actions)
            next_state, reward, done, _, _ = env.step(action)
            # Ensure reward is a float
            transition = (state, action, next_state, float(reward), done)

            # Check if the transition is in the transitions_train_set
            if transition in transitions_train_set:
                transitions.append(transition + (1,))  # adding probability

            state = next_state
            if len(transitions) >= num_steps:
                break

    return transitions


def run_sampling_regret_experiment_with_policy_evaluation(
    tau,
    seed,
    run_id,
    rows,
    cols,
    start_state,
    p_success,
    terminal_states,
    num_steps,
    epsilon,
    gamma,
    min_samples,
    batch_size,
    train_max_iterations,
    logger=None,
):

    if logger is None:
        logger = logging.getLogger(__name__)

    seed_everything(run_id)
    env = make_env(rows, cols, start_state, p_success, terminal_states, run_id)

    states = list(set([s for s, _ in env.mdp.keys()]))
    actions = list(set([a for _, a in env.mdp.keys()]))

    transitions_list = [(key[0], key[1], *value[0]) for key, value in env.mdp.items()]

    transitions_train, transitions_val = generate_train_test_split_with_valid_path(
        transitions_list=transitions_list,
        start_state=start_state,
        terminal_states=terminal_states,
        seed=run_id,
    )

    sampled_transitions_train = generate_transitions_observations(
        transitions_train, num_steps, tau=tau, min_samples=min_samples
    )

    ### Training
    input_size = len(states[0])
    output_size = len(actions)

    # Initialize the DQN
    seed_everything(seed)

    net_random_policy = QNET(input_size, output_size)

    loss_record_random_policy = train_net_with_neural_fitting(
        net_random_policy,
        sampled_transitions_train,
        states,
        actions,
        gamma,
        epsilon,
        batch_size,
        train_max_iterations,
        mode="mean",
        logger=logger,
    )

    bm_error = compute_validation_bellmans_error(
        net_random_policy,
        validation_transitions=transitions_val,
        error_mode="mean",
        gamma=gamma,
        logger=logger,
    )

    return loss_record_random_policy, bm_error


def run_baseline_random_policy_experiment(
    tau,
    seed,
    run_id,
    rows,
    cols,
    start_state,
    p_success,
    terminal_states,
    num_steps,
    epsilon,
    gamma,
    min_samples,
    batch_size,
    train_max_iterations,
    neural_fit_mode,
    logger=None,
):

    if logger is None:
        logger = logging.getLogger(__name__)

    seed_everything(run_id)
    env = make_env(rows, cols, start_state, p_success, terminal_states, run_id)

    states = list(set([s for s, _ in env.mdp.keys()]))
    actions = list(set([a for _, a in env.mdp.keys()]))
    random_policy = create_random_policy(states, actions)
    Q = {state: {action: 0 for action in actions} for state in states}
    Q_pi_random = random_policy_evaluation_q_stochastic(
        states, actions, random_policy, Q, env.mdp, gamma, epsilon
    )

    transitions_list = [(key[0], key[1], *value[0]) for key, value in env.mdp.items()]

    transitions_train, transitions_val = generate_train_test_split_with_valid_path(
        transitions_list=transitions_list,
        start_state=start_state,
        terminal_states=terminal_states,
        seed=run_id,
    )

    random_policy_transitions = generate_random_policy_transitions(
        transitions_train, num_steps, env, actions
    )

    seed_everything(seed)

    ### Training
    input_size = len(states[0])  # Or another way to represent the size of your input
    output_size = len(actions)

    # Initialize the DQN
    qnet_random_policy = QNET(input_size, output_size)

    loss_record = train_net_with_neural_fitting(
        qnet_random_policy,
        random_policy_transitions,
        Q_pi_random,
        states,
        actions,
        gamma,
        epsilon,
        batch_size=batch_size,
        max_iterations=train_max_iterations,
        frequency_scaling=False,
        mode=neural_fit_mode,
        logger=logger,
    )

    bm_error_validation = compute_validation_bellmans_error(
        qnet_random_policy,
        validation_transitions=transitions_val,
        error_mode=neural_fit_mode,
        gamma=gamma,
        logger=logger,
    )

    bm_error_train = compute_validation_bellmans_error(
        qnet_random_policy,
        validation_transitions=transitions_train,
        error_mode=neural_fit_mode,
        gamma=gamma,
        logger=logger,
    )

    return loss_record, bm_error_validation, bm_error_train


def run_distribution_correction_experiment(
    tau,
    seed,
    run_id,
    rows,
    cols,
    start_state,
    p_success,
    terminal_states,
    num_steps,
    gamma,
    min_samples,
    batch_size,
    train_max_iterations,
    neural_fit_mode,
    algorithm=None,
    logger=None,
):

    if logger is None:
        logger = logging.getLogger(__name__)

    if algorithm is None:
        logger.error("Algorithm must be provided")
        raise ValueError("Algorithm must be provided")

    if algorithm not in ["default", "adjusted_loss", "dataset_normed"]:
        logger.error("Algorithm must be default, adjusted_loss, or dataset_normed")
        raise ValueError("Algorithm must be default, adjusted_loss, or dataset_normed")

    seed_everything(run_id)
    env = make_env(rows, cols, start_state, p_success, terminal_states, run_id)

    states = list(set([s for s, _ in env.mdp.keys()]))
    actions = list(set([a for _, a in env.mdp.keys()]))
    transitions_list = [(key[0], key[1], *value[0]) for key, value in env.mdp.items()]

    transitions_train, transitions_val = generate_train_test_split_with_valid_path(
        transitions_list=transitions_list,
        start_state=start_state,
        terminal_states=terminal_states,
        seed=run_id,
    )

    train_dataset_transitions = generate_transitions_observations(
        transitions_train, num_steps, tau=tau, min_samples=min_samples
    )

    seed_everything(seed)
    input_size = len(states[0])
    output_size = len(actions)
    qnet = QNET(input_size, output_size)

    if algorithm == "default":
        loss_record = train_net_with_neural_fitting(
            qnet,
            train_dataset_transitions,
            gamma,
            batch_size,
            max_iterations=train_max_iterations,
            frequency_scaling=False,
            mode=neural_fit_mode,
            logger=logger,
        )

    if algorithm == "frequency_scaling":
        loss_record = train_net_with_neural_fitting(
            qnet,
            train_dataset_transitions,
            gamma,
            batch_size,
            max_iterations=train_max_iterations,
            frequency_scaling=True,
            mode=neural_fit_mode,
            logger=logger,
        )

    if algorithm == "dataset_normed":
        normalized_transitions_train = normalize_frequencies(train_dataset_transitions)
        loss_record = train_net_with_neural_fitting(
            qnet,
            normalized_transitions_train,
            gamma,
            batch_size,
            max_iterations=train_max_iterations,
            frequency_scaling=False,
            mode=neural_fit_mode,
            logger=logger,
        )

    bm_error = compute_validation_bellmans_error(
        qnet,
        validation_transitions=transitions_val,
        error_mode=neural_fit_mode,
        gamma=gamma,
    )

    return loss_record, bm_error


def build_graph_with_networkx(transitions, start_state, terminal_states):
    G = nx.DiGraph()
    G.add_node(start_state)
    for terminal_state in terminal_states.keys():
        G.add_node(terminal_state)
    for current_state, action, next_state, reward, done, prob in transitions:
        G.add_edge(current_state, next_state)
    return G


def check_path_existence_to_any_terminal(G, start_state, terminal_states):
    # Ensure the start state has at least one outgoing edge
    if not list(G.out_edges(start_state)):
        return False

    # Check for direct reachability to any terminal state
    terminal_reachable = any(
        list(G.in_edges(terminal_state)) for terminal_state in terminal_states.keys()
    )
    if not terminal_reachable:
        return False

    # Perform path existence check as before
    for terminal_state in terminal_states.keys():
        if nx.has_path(G, start_state, terminal_state):
            return True
    return False


def generate_train_test_split_with_valid_path(
    transitions_list, start_state, terminal_states, seed=None
):
    transitions_train, transitions_val = [], []
    found_valid_path = False
    attempts = 0
    max_attempts = 1000

    # Create a random number generator with the provided seed
    rng = np.random.default_rng(seed)

    while not found_valid_path and attempts < max_attempts:
        # Generate a new seed for each split attempt
        split_seed = rng.integers(low=0, high=10000)

        transitions_train, transitions_val = train_test_split(
            transitions_list, test_size=0.2, random_state=split_seed
        )
        G = build_graph_with_networkx(transitions_train, start_state, terminal_states)
        if check_path_existence_to_any_terminal(G, start_state, terminal_states):
            found_valid_path = True
        else:
            attempts += 1  # Increment the attempt counter

    if not found_valid_path:
        raise ValueError(
            "Could not find a valid path to any terminal state after {} attempts. Check setup.".format(
                max_attempts
            )
        )

    return transitions_train, transitions_val


def run_dqn_distribution_correction_experiment(
    config,
    logger=None,
):

    if logger is None:
        logger = logging.getLogger(__name__)

    algorithm = config.get("algorithm")
    if algorithm is None:
        logger.error("Algorithm must be provided")
        raise ValueError("Algorithm must be provided")

    if algorithm not in ["default", "adjusted_loss", "dataset_normed"]:
        logger.error("Algorithm must be default, adjusted_loss, or dataset_normed")
        raise ValueError("Algorithm must be default, adjusted_loss, or dataset_normed")

    # seed_everything(run_id)
    # env = make_env(rows, cols, start_state, p_success, terminal_states, run_id)

    # states = list(set([s for s, _ in env.mdp.keys()]))
    # actions = list(set([a for _, a in env.mdp.keys()]))
    # transitions_list = [(key[0], key[1], *value[0]) for key, value in env.mdp.items()]

    seed = config.get("seed")
    seed_everything(seed)

    agent = setup_dqn_agent(
        config=config,
        logger=logger,
    )
    
    logger.info(f"Agent parametrization: {agent.get_settings()}")

    transitions_list = [
        (key[0], key[1], *value[0]) for key, value in agent.train_env.mdp.items()
    ]

    experiment_data = []
    for i in range(1, config["train_max_iterations"] + 1):
        agent.train(i)

        bm_error_validation = compute_validation_bellmans_error(
            agent.target_model,
            validation_transitions=transitions_list,
            error_mode=config["neural_fit_mode"],
            gamma=agent.gamma,
            logger=logger,
        )

        rb_entropy = agent.replay_buffer.calculate_buffer_entropy()

        normalized_rb = agent.replay_buffer.normalize_replay_buffer()
        normalized_rb_entropy = normalized_rb.calculate_buffer_entropy()

        examples = [
            (transition[0], transition[1]) for transition in agent.replay_buffer.buffer
        ]
        example_strings = [f"{state}_{action}" for state, action in examples]
        unique_examples, counts = np.unique(example_strings, return_counts=True)

        experiment_data.append(
            {
                "epoch": i,
                "bellman_error": bm_error_validation,
                "replay_buffer_entropy": rb_entropy,
                "normalized_replay_buffer_entropy": normalized_rb_entropy,
                "nr_unique_examples": len(unique_examples),
            }
        )

    return experiment_data


def setup_dqn_agent(
    config,
    logger,
):
    logger.info(f"Starting experiment: {config['full_title']}")

    rows = config["rows"]
    cols = config["cols"]
    start_state = config["start_state"]
    p_success = config["p_success"]
    terminal_states = config["terminal_states"]
    run_id = config["run_id"]
    episode_length_limit = config.get("episode_length_limit")
    walls = set(config["walls"]) if config.get("walls") else None

    if config["algorithm"] == "dataset_normed":
        config["normalize_replay_buffer_freq"] = True

    ### Setup environments ###
    train_env = make_env(
        rows,
        cols,
        start_state,
        p_success,
        terminal_states,
        run_id,
        walls=walls,
        episode_length_limit=episode_length_limit,
    )
    validation_env = make_env(
        rows,
        cols,
        start_state,
        p_success,
        terminal_states,
        run_id,
        episode_length_limit=episode_length_limit,
    )

    ### Setup output and loading paths ###

    experiment_agent = AgentDQN(
        train_env=train_env,
        validation_env=validation_env,
        experiment_output_folder=config["out_dir"],
        experiment_name=config["experiment"],
        resume_training_path=None,
        save_checkpoints=True,
        logger=logger,
        config=config,
    )

    return experiment_agent
