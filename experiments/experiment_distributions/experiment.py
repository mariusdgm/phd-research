import os
import sys
import time
from liftoff import parse_opts
from argparse import Namespace

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

from experiments.experiment_utils import setup_logger


def run(opts: Namespace) -> None:
    logger = setup_logger(
        opts.full_title, log_file=os.path.join(opts.out_dir, "experiment_log.log")
    )
    
    

    return True


def main():
    opts = parse_opts()
    run(opts)


if __name__ == "__main__":
    main()