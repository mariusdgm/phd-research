"""Rerun experiments but instead of NFQI and NFPE, run DQN

"""

import os
import sys
import time
import random
import pandas as pd
import yaml
import csv
from liftoff import parse_opts
from argparse import Namespace

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(root_dir)

from common.src.distribution_src import run_dqn_distribution_correction_experiment
from common.src.experiment_utils import (
    setup_logger,
    cleanup_file_handlers,
    namespace_to_dict,
)

def log_types(logger, structure, prefix=''):
    if isinstance(structure, dict):
        for key, value in structure.items():
            logger.info(f"{prefix}{key}: {type(value).__name__}")
            log_types(logger, value, prefix + '  ')
    elif isinstance(structure, list):
        for i, value in enumerate(structure):
            logger.info(f"{prefix}[{i}]: {type(value).__name__}")
            log_types(logger, value, prefix + '  ')
    else:
        logger.info(f"{prefix}{type(structure).__name__}")

# TODO: modify epsilon greedy to reach exploitation after 33%
# TODO: add a parameter for skewness (entropy) for the replay buffer to check distribution

def run(opts: Namespace) -> None:
    logger = setup_logger(
        opts.full_title, log_file=os.path.join(opts.out_dir, "experiment_log.log")
    )
    opts.seed = random.randint(0, 2**32 - 1) if opts.seed is None else opts.seed
    opts_dict = namespace_to_dict(opts)
    
   

    with open(os.path.join(opts.out_dir, "post_cfg.yaml"), "w") as f:
        # Use PyYAML to write the dictionary to a YAML file
        yaml.dump(opts_dict, f)

    logger.info(
        f"Starting experiment {opts.title}, seed {opts.run_id}, out_dir {opts.out_dir}"
    )

    # TODO: implement dqn experiment with parametrization
    bm_error_records = run_dqn_distribution_correction_experiment(
        config=opts_dict,
        logger=logger,
    )

    logger.info(
        f"Finished experiment {opts.title}, seed {opts.run_id}, out_dir {opts.out_dir}"
    )
    
    df_loss = pd.DataFrame(
        bm_error_records, columns=["epoch", "bellman_error"]
    )

    # Saving the loss record to a CSV file
    loss_record_path = os.path.join(opts.out_dir, "bellman_errors.csv")
    df_loss.to_csv(loss_record_path, index=False)
    
    cleanup_file_handlers(logger)

    return True


def main():
    opts = parse_opts()
    run(opts)


if __name__ == "__main__":
    main()
