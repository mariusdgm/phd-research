import os, sys

root_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
sys.path.append(root_dir)

import numpy as np
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

from rl_envs_forge.envs.grid_world.grid_world import GridWorld
from rl_envs_forge.envs.grid_world.grid_world import Action

from common.src.distribution_src import (
    train_net_with_neural_fitting, 
    compute_validation_bellmans_error, 
    make_env, 
    QNET, 
    generate_train_test_split_with_valid_path, 
    generate_transitions_observations,
    normalize_frequencies
    )
from common.src.utils import create_random_policy
from common.src.policy_iteration import random_policy_evaluation_q_stochastic
from experiments.experiment_utils import seed_everything

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LinearLR
from collections import Counter

import networkx as nx


def run_sampling_dqn_experiment(
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
    dqn_config,
    logger=None,
)

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

    train_dataset_transitions = generate_transitions_observations(
        transitions_train, num_steps, tau=tau, min_samples=min_samples
    )
    # train_dataset_transitions = generate_random_policy_transitions(
    #     transitions_train, num_steps, env, actions, seed, logger
    # )

    ### Training
    input_size = len(states[0])  # Or another way to represent the size of your input
    output_size = len(actions)
    seed_everything(seed)

    # Initialize and train network with original loss
    qnet = QNET(input_size, output_size)
    loss_record_random_policy = train_dqn(
        qnet,
        train_dataset_transitions,
        Q_pi_random,
        states,
        actions,
        gamma,
        epsilon,
        batch_size,
        max_iterations=train_max_iterations,
        frequency_scaling=False,
        mode=neural_fit_mode,
        logger=logger,
    )

    # Initialize and train network with original loss
    qnet_adjusted_loss = QNET(input_size, output_size)
    loss_record_random_policy_adjusted = train_dqn_with_adjusted_loss(
        qnet_adjusted_loss,
        train_dataset_transitions,
        Q_pi_random,
        states,
        actions,
        gamma,
        epsilon,
        batch_size,
        max_iterations=train_max_iterations,
        frequency_scaling=True,
        mode=neural_fit_mode,
        logger=logger,
    )
    
    normalized_transitions_train = normalize_frequencies(train_dataset_transitions)
    qnet_dataset_normed = QNET(input_size, output_size)
    loss_record_dataset_normed = train_net_with_neural_fitting(
        qnet,
        normalized_transitions_train,
        Q_pi_random,
        states,
        actions,
        gamma,
        epsilon,
        batch_size,
        max_iterations=train_max_iterations,
        frequency_scaling=False,
        mode=neural_fit_mode,
        logger=logger,
    )
    
    bm_error = compute_validation_bellmans_error(
        qnet, validation_transitions=transitions_val, error_mode=neural_fit_mode, gamma=gamma
    )

    bm_error_adjusted = compute_validation_bellmans_error(
        qnet_adjusted_loss,
        validation_transitions=transitions_val,
        error_mode=neural_fit_mode,
        gamma=gamma,
        logger=logger,
    )
    
    bm_dataset_normed = compute_validation_bellmans_error(
        qnet_dataset_normed,
        validation_transitions=transitions_val,
        error_mode=neural_fit_mode,
        gamma=gamma,
        logger=logger,
    )

    loss = {
        "qnet_original": loss_record_random_policy,
        "qnet_adjusted_loss": loss_record_random_policy_adjusted,
        "qnet_dataset_normed": loss_record_dataset_normed,
    }

    bm_error = {"qnet_original": bm_error, 
                "qnet_adjusted_loss": bm_error_adjusted,
                "qnet_dataset_normed": bm_dataset_normed
                }

    return loss, bm_error

def train_dqn_with_adjusted_loss():
    pass

def train_dqn():
    pass