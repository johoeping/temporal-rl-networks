"""
Python script for an example training run with MLP.
"""

import os
import sys
import subprocess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.logger_setup import get_logger


logger = get_logger(__name__)

# Define paths
CONFIG_PATH = "configs/example_config_mlp.yaml"
OUTPUT_DIR = "data/example_output/mlp"
RUN_DIR = os.path.join(OUTPUT_DIR, 'run_walker2d-v5_mlp_arch16-16_2025')

# Set up environment with PYTHONPATH to include the project root
env = os.environ.copy()
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
env['PYTHONPATH'] = project_root + os.pathsep + env.get('PYTHONPATH', '')

# Helper function to run shell commands
def run_command(command, description):
    try:
        subprocess.run(command, check=True, shell=True, env=env)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during {description}: {e} Check logs for details. Check if the config is set up correctly.")
        sys.exit()

# Step 1: Execute training
run_command(f"python run.py --config {CONFIG_PATH} --output-dir {OUTPUT_DIR}", "Training")

# Step 2: Collect data
run_command(f"python src/metrics/collect_data.py {CONFIG_PATH}", "Data collection")

# Step 3: Analyze cycles
run_command(f"python src/metrics/analyze_cycles.py {os.path.join(RUN_DIR, 'cycle_analysis', 'actions', 'dataset_actions.csv')} --output {os.path.join(RUN_DIR, 'cycle_analysis', 'actions')}", "Cycle analysis (actions)")
run_command(f"python src/metrics/analyze_cycles.py {os.path.join(RUN_DIR, 'cycle_analysis', 'observations', 'dataset_observations.csv')} --output {os.path.join(RUN_DIR, 'cycle_analysis', 'observations')}", "Cycle analysis (observations)")