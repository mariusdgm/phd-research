import os
import sys
import time
import random
import pandas as pd
import yaml
from liftoff import parse_opts
from argparse import Namespace

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(root_dir)

from experiment_src import run_sampling_regret_experiment
from experiments.experiment_utils import (
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
    opts.terminal_states = namespace_to_dict(opts.terminal_states)
    opts_dict = vars(opts)

    with open(os.path.join(opts.out_dir, "post_cfg.yaml"), "w") as f:
        # Use PyYAML to write the dictionary to a YAML file
        yaml.dump(opts_dict, f)

    logger.info(
        f"Starting experiment {opts.title}, seed {opts.run_id}, out_dir {opts.out_dir}"
    )

    loss_record, bm_error = run_sampling_regret_experiment(
        tau=opts.tau,
        seed=opts.seed,
        run_id=opts.run_id,
        rows=opts.rows,
        cols=opts.cols,
        start_state=opts.start_state,
        p_success=opts.p_success,
        terminal_states=opts.terminal_states,
        num_steps=opts.num_steps,
        epsilon=opts.epsilon,
        gamma=opts.gamma,
        batch_size=opts.batch_size,
        min_samples=opts.min_samples,
        train_max_iterations=opts.train_max_iterations,
        logger=logger,
    )

    logger.info(
        f"Finished experiment {opts.title}, seed {opts.run_id}, out_dir {opts.out_dir}"
    )

    # Convert results to DataFrame (if not already in a suitable format)
    df_loss = pd.DataFrame(
        loss_record, columns=["epoch", "total_loss"]
    )

    # Saving the loss record to a CSV file
    loss_record_path = os.path.join(opts.out_dir, "loss_record.csv")
    df_loss.to_csv(loss_record_path, index=False)

    # Save Bellman error to a text file or as part of another CSV
    bm_error_path = os.path.join(opts.out_dir, "bellman_error.txt")
    with open(bm_error_path, "w") as f:
        f.write(str(bm_error))

    logger.info(f"Saved loss record to {loss_record_path}")
    logger.info(f"Saved Bellman error to {bm_error_path}")

    return True


def main():
    opts = parse_opts()
    run(opts)


if __name__ == "__main__":
    main()
