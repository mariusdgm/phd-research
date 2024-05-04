import os, sys
import random

root_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
sys.path.append(root_dir)

import numpy as np
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

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
from common.src.experiment_utils import seed_everything, cleanup_file_handlers, namespace_to_dict


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LinearLR
from collections import Counter

import networkx as nx
import logging


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
    
    transitions_list = [(key[0], key[1], *value[0]) for key, value in env.mdp.items()]

    transitions_train, transitions_val = generate_train_test_split_with_valid_path(
        transitions_list=transitions_list,
        start_state=start_state,
        terminal_states=terminal_states,
        seed=run_id,
    )

    # train_dataset_transitions = generate_transitions_observations(
    #     transitions_train, num_steps, tau=tau, min_samples=min_samples
    # )
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
        transitions_train,
        states,
        actions,
        gamma,
        epsilon,
        batch_size,
        max_iterations=train_max_iterations,
        frequency_scaling=False,
        mode=neural_fit_mode,
        adjusted_loss=False,
        logger=logger,
    )

    # Initialize and train network with original loss
    qnet_adjusted_loss = QNET(input_size, output_size)
    loss_record_random_policy_adjusted = train_dqn(
        qnet_adjusted_loss,
        transitions_train
        states,
        actions,
        gamma,
        epsilon,
        batch_size,
        max_iterations=train_max_iterations,
        frequency_scaling=False,
        mode=neural_fit_mode,
        adjusted_loss=True,
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

        
    
def train_dqn(config, logger):
    try:
        seed = int(os.path.basename(config["out_dir"]))

        seed_everything(seed)

        logs_file = os.path.join(config["out_dir"], "experiment_log.log")

        logger.info(f"Starting experiment: {config['full_title']}")

        ### Setup environments ###
        train_env = env = make_env(rows, cols, start_state, p_success, terminal_states, run_id)
        validation_env = env = make_env(rows, cols, start_state, p_success, terminal_states, run_id)

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
            enable_tensorboard_logging=False,
        )

        experiment_agent.train(train_epochs=config["epochs_to_train"])

        logger.info(
            f'Finished training experiment: {config["full_title"]}, seed: {config["seed"]}'
        )

        cleanup_file_handlers(experiment_logger=logger)

        return True

    except Exception as exc:
        # Capture the stack trace along with the exception message
        error_info = traceback.format_exc()

        # Log this information using your logger, if it's available
        logger.error("An error occurred: %s", error_info)

        # Return the error info so it can be collected by the parent process
        return error_info