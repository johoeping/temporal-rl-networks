"""
Python script for an example training run with CycleNet training on an offline MLP expert.
"""

import os
import sys
import subprocess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.logger_setup import get_logger


logger = get_logger(__name__)

# Define paths
CONFIG_PATH_MLP = "configs/example_config_mlp.yaml"
OUTPUT_DIR_MLP = "data/example_output/mlp"
RUN_DIR_MLP = os.path.join(OUTPUT_DIR_MLP, 'run_walker2d-v5_mlp_arch16-16_2025')

CONFIG_PATH_CYCLENET_OFFLINE = "configs/example_config_cyclenet_offline.yaml"
OUTPUT_DIR_CYCLENET_OFFLINE = "data/example_output/cyclenet_offline"
RUN_DIR_CYCLENET_OFFLINE = os.path.join(OUTPUT_DIR_CYCLENET_OFFLINE, 'run_walker2d-v5_cyclenet_seq16_cycle16_linear_dmodel16_revinfalse_2025')

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

# -- MLP EXPERT TRAINING --

# Step 1: Execute training
run_command(f"python run.py --config {CONFIG_PATH_MLP} --output-dir {OUTPUT_DIR_MLP}", "Training")

# Step 2: Collect data
run_command(f"python src/metrics/collect_data.py {CONFIG_PATH_MLP}", "Data collection")

# Step 3: Analyze cycles
run_command(f"python src/metrics/analyze_cycles.py {os.path.join(RUN_DIR_MLP, 'cycle_analysis', 'actions', 'dataset_actions.csv')} --output {os.path.join(RUN_DIR_MLP, 'cycle_analysis', 'actions')} --is-cyclenet", "Cycle analysis (actions)")
run_command(f"python src/metrics/analyze_cycles.py {os.path.join(RUN_DIR_MLP, 'cycle_analysis', 'observations', 'dataset_observations.csv')} --output {os.path.join(RUN_DIR_MLP, 'cycle_analysis', 'observations')} --is-cyclenet", "Cycle analysis (observations)")


# -- OFFLINE CYCLENET TRAINING --

# Step 1: Execute training
run_command(f"python run.py --config {CONFIG_PATH_CYCLENET_OFFLINE} --output-dir {OUTPUT_DIR_CYCLENET_OFFLINE}", "Training")

# Step 2: Collect data
run_command(f"python src/metrics/collect_data.py {CONFIG_PATH_CYCLENET_OFFLINE}", "Data collection")

# Step 3: Analyze cycles
run_command(f"python src/metrics/analyze_cycles.py {os.path.join(RUN_DIR_CYCLENET_OFFLINE, 'cycle_analysis', 'actions', 'dataset_actions.csv')} --output {os.path.join(RUN_DIR_CYCLENET_OFFLINE, 'cycle_analysis', 'actions')} --is-cyclenet", "Cycle analysis (actions)")
run_command(f"python src/metrics/analyze_cycles.py {os.path.join(RUN_DIR_CYCLENET_OFFLINE, 'cycle_analysis', 'observations', 'dataset_observations.csv')} --output {os.path.join(RUN_DIR_CYCLENET_OFFLINE, 'cycle_analysis', 'observations')} --is-cyclenet", "Cycle analysis (observations)")
