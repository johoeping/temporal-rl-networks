# Temporal RL Networks

A research framework for comparing temporal dependency modeling approaches in Deep Reinforcement Learning. This project integrates the [CycleNet](https://github.com/ACAT-SCUT/CycleNet) architecture with traditional MLP and LSTM networks for Reinforcement Learning in periodic environments.

Developed as part of a bachelor's thesis on *"Temporal Dependency Modeling in Deep Reinforcement Learning: Comparing CycleNet to Established Architectures"*, this framework provides the basis for exploring whether CycleNet's explicit cyclic mechanisms can effectively capture temporal patterns in locomotion tasks with potentially rhythmic structures, such as walking gaits or periodic movement behaviors.

## 🚀 Features

### Neural Network Architectures
- **[CycleNet](https://github.com/ACAT-SCUT/CycleNet)**: Temporal modeling with learnable recurrent cycles
- **LSTM**: Networks with memory for temporal dependencies as baseline
- **MLP**: Traditional feedforward networks as baseline

### Training and Analysis
- **Training Modes**: Online (PPO) and Offline (Behavioral Cloning)
- **Environments**: Gymnasium MuJoCo continuous control tasks (Walker2d, Hopper, Ant, etc.)
- **Temporal Wrappers**: Cycle index and sequence handling for CycleNet
- **Cycle Analysis**: Periodicity detection using Autocorrelation, FFT, and Welch's Periodogram

## 🛠️ Installation

### Prerequisites
- Python 3.10
- CUDA-compatible GPU (optional)

### Setup
```bash
# Clone the repository
git clone https://github.com/johoeping/temporal-rl-networks.git
cd temporal-rl-networks

# Create and activate conda environment
conda env create -f environment.yaml
conda activate temporal_rl_env

# Verify environment installation
python -c "import gymnasium as gym; env = gym.make('Walker2d-v5'); print('Setup successful!')"
```

## 🎯 Quick Start

### Basic Training
```bash
# Train a CycleNet model
python run.py --config configs/example_config_cyclenet.yaml --output-dir data/example_output/cyclenet

# Train an LSTM model
python run.py --config configs/example_config_lstm.yaml --output-dir data/example_output/lstm

# Train an MLP model
python run.py --config configs/example_config_mlp.yaml --output-dir data/example_output/mlp
```

### Data Collection and Analysis
The framework includes specialized tools for trajectory data collection and temporal pattern analysis (this always requires a trained model):

#### Collect Trajectory Data
```bash
# Collect data from a trained model
python src/metrics/collect_data.py configs/example_config_cyclenet.yaml
```
Creates datasets from trained RL agents with statistical analysis and CSV export for further processing.

#### Analyze Periodic Patterns  
```bash
# Analyze cycles in trajectory data
python src/metrics/analyze_cycles.py data/example_output/mlp/cycle_analysis/actions/dataset_actions.csv --output data/example_output/mlp/cycle_analysis/actions
```
Detects periodicity using Autocorrelation, FFT, and Periodogram methods with comprehensive visualization.

### Using Pipelines
Execute complete end-to-end workflows that combine training, data collection, and analysis:

```bash
# CycleNet Pipeline: Train a CycleNet agent, collect trajectories, and analyze cycles
python pipelines/pipeline_cyclenet.py

# LSTM Pipeline: Train an LSTM agent, collect trajectories, and analyze cycles
python pipelines/pipeline_lstm.py

# MLP Pipeline: Train an MLP agent, collect trajectories, and analyze cycles
python pipelines/pipeline_mlp.py

# CycleNet Offline Pipeline:
# 1. Train an MLP expert agent, collect trajectories, and analyze cycles
# 2. Train a CycleNet agent offline based on the expert data, collect trajectories, and analyze cycles
python pipelines/pipeline_cyclenet_offline.py
```

Each pipeline automatically:
- Trains the respective model using PPO
- Collects trajectory data from the trained agent
- Performs cycle analysis and baseline evaluation
- Saves results to `data/example_output/` with structured output files

## ⚙️ Configuration

### Basic Configuration Structure
```yaml
# Environment
environment:
  name: "Walker2d-v5"
  noise_std: 0.0

# Model Architecture  
model:
  network_type: "cyclenet"  # "cyclenet", "lstm", "mlp"
  cyclenet:
    seq_len: 16
    cycle: 16
    model_type: "linear"    # "linear" or "mlp"

# Training
training:
  mode: "online"            # "online" or "offline"
  online:
    total_timesteps: 100000
    learning_rate: 0.0003
```

See `configs/` directory for complete example configurations.

### Overriding Configuration Values

You can specify a configuration file using the `--config` argument and override individual configuration values directly via command-line arguments. For example:

```bash
python run.py \
    --config configs/example_config_cyclenet.yaml \
    --output-dir data/example_output/cyclenet \
    --env-name Hopper-v5 \
    --online-total-timesteps 200000
```

In this example, the `env-name` and `online-total-timesteps` values override the corresponding values in the configuration file.

You can use the `--help` flag to display all available options and their descriptions.

## 🙏 Acknowledgments

This research builds upon:

- **[CycleNet](https://github.com/ACAT-SCUT/CycleNet)**: Time series forecasting framework using Residual Cycle Forecasting (RCF) technique by Lin et al. (NeurIPS 2024). The `CycleNet/` directory contains core components (models, LICENSE, README) from the original CycleNet repository
- **[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)**: RL algorithm implementations  
- **[Gymnasium](https://github.com/Farama-Foundation/Gymnasium)**: RL environment interface

---

**License**: See [LICENSE](LICENSE) for details.
