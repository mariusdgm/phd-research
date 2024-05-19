import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

import subprocess
import threading
import shutil
import ast
from argparse import Namespace

from typing import List

import logging
from logging.handlers import RotatingFileHandler

import pandas as pd
from sklearn.model_selection import train_test_split

from rl_envs_forge.envs.grid_world.grid_world import GridWorld
# from overfitting.src.utils import extract_V_from_Q, create_random_policy

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


def setup_logger(name, log_file=None, level=logging.INFO):
    """Set up a logger that logs to the console and optionally to a file."""

    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create a console handler
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # If a log file path is provided, set up file handler
    if log_file is not None:
        file_handler = RotatingFileHandler(
            log_file, maxBytes=1024 * 1024 * 5, backupCount=3
        )  # 5MB per file, with 3 backups
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger

def cleanup_file_handlers(experiment_logger=None):
    """Cleans up all handlers of experiment_logger logger instance.

    Args:
        experiment_logger (Logger, optional): If experiment_logger is None, then all loggers will be cleaned up.
        Defaults to None.
    """
    # Get all active loggers
    if experiment_logger:
        for handler in experiment_logger.handlers:
            handler.close()
            experiment_logger.removeHandler(handler)
    
    else:
        logger = logging.getLogger()
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)
        


def run_cli_command(command: List[str], cwd: str) -> subprocess.CompletedProcess:
    # Run the CLI command and return the output
    command = " ".join(command)  # Need to join command so it can run on Linux systems
    return subprocess.run(
        command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )


def run_cli_command_non_blocking(command: List[str], cwd: str):
    """Run the CLI command in a non-blocking manner and return the thread."""
    command = " ".join(command)  # Need to join command so it can run on Linux systems

    def target():
        subprocess.run(
            command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
        )

    thread = threading.Thread(target=target)
    thread.start()
    return thread


def clean_up_directory(directory_path):
    if os.path.exists(directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")


def seed_everything(seed):
    """
    Set the seed on everything I can think of.
    Hopefully this should ensure reproducibility.

    Credits: Florin
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def is_tuple_string(s):
    try:
        # Use ast.literal_eval to safely evaluate if the string is a tuple
        evaluated = ast.literal_eval(s)
        return isinstance(evaluated, tuple)
    except:
        return False

def convert_tuple_string_to_tuple(key):
    try:
        return ast.literal_eval(key)
    except (ValueError, SyntaxError):
        return key

def namespace_to_dict(namespace):
    if isinstance(namespace, Namespace):
        return {key: namespace_to_dict(value) for key, value in vars(namespace).items()}
    elif isinstance(namespace, list):
        return [namespace_to_dict(item) for item in namespace]
    elif isinstance(namespace, dict):
        return {convert_tuple_string_to_tuple(key): namespace_to_dict(value) for key, value in namespace.items()}
    elif isinstance(namespace, str) and is_tuple_string(namespace):
        return ast.literal_eval(namespace)
    else:
        return namespace
