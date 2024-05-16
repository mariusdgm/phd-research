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
    convert_from_string,
    namespace_to_dict,
)


def run(opts: Namespace) -> None:
    logger = setup_logger(
        opts.full_title, log_file=os.path.join(opts.out_dir, "experiment_log.log")
    )
    opts.seed = random.randint(0, 2**32 - 1) if opts.seed is None else opts.seed
    opts.start_state = convert_from_string(opts.start_state)
    opts_dict = namespace_to_dict(opts)

    with open(os.path.join(opts.out_dir, "post_cfg.yaml"), "w") as f:
        # Use PyYAML to write the dictionary to a YAML file
        yaml.dump(opts_dict, f)

    logger.info(
        f"Starting experiment {opts.title}, seed {opts.run_id}, out_dir {opts.out_dir}"
    )

    # TODO: implement dqn experiment with parametrization
    run_dqn_distribution_correction_experiment(
        config=opts_dict,
        logger=logger,
    )

    logger.info(
        f"Finished experiment {opts.title}, seed {opts.run_id}, out_dir {opts.out_dir}"
    )

    return True


def main():
    opts = parse_opts()
    run(opts)


if __name__ == "__main__":
    main()
