#!/usr/bin/env python3
"""
Script to reproduce the results published in the paper.

This script reads a YAML file containing experiment definitions
corresponding to the paper's results, checks out the config/ folder 
from specified commits, and runs DVC experiments to replicate 
the published findings.
"""

import argparse
import logging
from pathlib import Path
import subprocess
import sys

import yaml

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(command, check=True):
    """Run a shell command and return the result."""
    logger.info(f"Running: {' '.join(command)}")
    try:
        result = subprocess.run(
            command,
            check=check,
            capture_output=True,
            text=True
        )
        if result.stdout:
            logger.debug(f"STDOUT: {result.stdout}")
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        if e.stderr:
            logger.error(f"STDERR: {e.stderr}")
        raise


def checkout_config_from_commit(commit_hash):
    """Checkout the config/ folder from a specific commit."""
    logger.info(f"Checking out config/ from commit {commit_hash}")
    run_command(["git", "checkout", commit_hash, "--", "config/"])


def run_dvc_experiment(experiment_name, additional_params=None):
    """Run a DVC experiment with the given name and optional parameters."""
    logger.info(f"Running DVC experiment: {experiment_name}")

    cmd = ["dvc", "exp", "run", "--name", experiment_name]

    if additional_params:
        for param, value in additional_params.items():
            cmd.extend(["--set-param", f"{param}={value}"])

    run_command(cmd)


def restore_git_state():
    """Restore git to a clean state."""
    logger.info("Restoring git state")
    run_command(["git", "reset", "--hard", "HEAD"])


def load_experiments_config(config_file):
    """Load the experiments configuration from YAML file."""
    logger.info(f"Loading experiments configuration from {config_file}")
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the results published in the paper by running experiments from configuration file"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="YAML file containing experiment configurations"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually running experiments"
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_experiments_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    experiments = config.get('experiments', [])
    if not experiments:
        logger.error("No experiments found in configuration file")
        sys.exit(1)

    logger.info(f"Found {len(experiments)} experiments to run")

    # Store original git state
    original_branch = run_command(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        check=False
    ).stdout.strip()

    try:
        for i, experiment in enumerate(experiments, 1):
            logger.info(f"Processing experiment {i}/{len(experiments)}")

            # Validate experiment configuration
            if 'name' not in experiment:
                logger.error(f"Experiment {i} missing required 'name' field")
                continue

            if 'commit' not in experiment:
                logger.error(f"Experiment {i} missing required 'commit' field")
                continue

            experiment_name = experiment['name']
            commit_hash = experiment['commit']
            additional_params = experiment.get('params', {})

            logger.info(f"Experiment: {experiment_name}")
            logger.info(f"Commit: {commit_hash}")
            if additional_params:
                logger.info(f"Additional params: {additional_params}")

            if args.dry_run:
                logger.info(
                    "DRY RUN: Would checkout config and run experiment")
                continue

            try:
                # Checkout config from specified commit
                checkout_config_from_commit(commit_hash)

                # Run DVC experiment
                run_dvc_experiment(experiment_name, additional_params)

                logger.info(
                    f"Successfully completed experiment: {experiment_name}")

            except Exception as e:
                logger.error(
                    f"Failed to run experiment {experiment_name}: {e}")
                continue

            finally:
                # Restore git state after each experiment
                try:
                    restore_git_state()
                except Exception as e:
                    logger.warning(f"Failed to restore git state: {e}")

    finally:
        # Ensure we're back to original branch
        if original_branch and original_branch != "HEAD":
            try:
                run_command(["git", "checkout", original_branch])
                logger.info(f"Restored to original branch: {original_branch}")
            except Exception as e:
                logger.warning(f"Failed to restore original branch: {e}")

    logger.info("All experiments completed")


if __name__ == "__main__":
    main()
