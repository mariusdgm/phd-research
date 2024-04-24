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

from experiment_src import run_adjusted_loss_baseline_experiment
from experiments.experiment_utils import (
    setup_logger,
    seed_everything,
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

    loss_records, bm_error = run_adjusted_loss_baseline_experiment(
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
        min_samples=opts.min_samples,
        batch_size=opts.batch_size,
        train_max_iterations=opts.train_max_iterations,
        neural_fit_mode=opts.neural_fit_mode,
        logger=logger,
    )

    logger.info(
        f"Finished experiment {opts.title}, seed {opts.run_id}, out_dir {opts.out_dir}"
    )

    df_loss_list = []
    for model_key, loss_record in loss_records.items():
        # Create a DataFrame for the current model's loss record
        df_temp = pd.DataFrame(
            loss_record, columns=["epoch", "total_loss", "scheduler_lr"]
        )
        df_temp["model"] = model_key  # Dynamically set the model name based on the key
        df_loss_list.append(df_temp)

    # Concatenating all the DataFrames in the list
    df_loss_combined = pd.concat(df_loss_list)

    # Saving the combined DataFrame to a CSV file
    loss_record_path = os.path.join(opts.out_dir, "loss_record_combined.csv")
    df_loss_combined.to_csv(loss_record_path, index=False)

    # Directly convert the bm_error dictionary to a DataFrame
    df_bm_error = pd.DataFrame(
        list(bm_error.items()), columns=["model", "bellman_error"]
    )

    # Saving the Bellman error DataFrame to a CSV file
    bm_error_path = os.path.join(opts.out_dir, "bellman_error_combined.csv")
    df_bm_error.to_csv(bm_error_path, index=False)

    logger.info(f"Saved loss record to {loss_record_path}")
    logger.info(f"Saved Bellman error to {bm_error_path}")

    return True


def main():
    opts = parse_opts()
    run(opts)


if __name__ == "__main__":
    main()
