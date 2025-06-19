# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Rules

- Only write code in the /claude/ dir. you can reference files anywhere, but keep your changes within that dir. if you have to make changes to other files, copy them into the claude dir and then make changes. 

- every time you finish a logical piece of work, summarize what you did and any tradeoffs you made in a claude/WORKLOG.md file. append it with an EST timestamp.

## Overview

OpenPI is a repository containing open-source models and packages for robotics from Physical Intelligence. It provides two main model types:
- **π₀ (pi0)**: Flow-based diffusion vision-language-action model (VLA)
- **π₀-FAST (pi0_fast)**: Autoregressive VLA based on FAST action tokenizer

The repository supports both base models for fine-tuning and pre-trained models for specific robot platforms (ALOHA, DROID, Libero).

## Development Commands

### Environment Setup
this should already be done for you.

```bash
# Clone with submodules
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
git submodule update --init --recursive

# Setup Python environment using uv
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
source .venv/bin/activate
```

### Code Quality
```bash
# Linting and formatting
ruff check .
ruff format .

# Pre-commit hooks
pre-commit install
pre-commit run --all-files

# Run tests
pytest
```

### Training
```bash
# Compute normalization statistics (required before training)
uv run scripts/compute_norm_stats.py --config-name <config_name>

# Run training with optimal GPU memory usage
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config_name> --exp-name=<experiment_name> --overwrite
```

### Inference
```bash
# Serve a policy (for remote inference)
uv run scripts/serve_policy.py policy:checkpoint --policy.config=<config_name> --policy.dir=<checkpoint_path>

# Test with simple client (no robot required)
uv run examples/simple_client/main.py --env <ENV_TYPE>
```

## Architecture

### Core Components

**Models** (`src/openpi/models/`):
- `model.py`: Base model definitions and data structures (`Observation`, `Actions`)
- `pi0.py`: Flow-based diffusion model implementation
- `pi0_fast.py`: Autoregressive model implementation
- `gemma.py`, `gemma_fast.py`: Gemma-based language model components
- `siglip.py`, `vit.py`: Vision components
- `tokenizer.py`: Text tokenization utilities

**Policies** (`src/openpi/policies/`):
- `policy.py`: Base Policy class wrapping models for inference
- `*_policy.py`: Robot-specific policy implementations (ALOHA, DROID, Libero)
- `policy_config.py`: Configuration and factory functions for policies

**Training** (`src/openpi/training/`):
- `config.py`: Training configurations and data pipeline setup
- `data_loader.py`: LeRobot dataset integration and data loading
- `train.py`: Main training script with logging and checkpointing
- `optimizer.py`: Optimization utilities
- `checkpoints.py`: Model checkpointing utilities

**Data Processing**:
- `transforms.py`: Data transformation pipeline
- `shared/normalize.py`: Normalization utilities
- `shared/image_tools.py`: Image processing utilities

### Data Flow

1. **Raw Data**: LeRobot datasets containing robot demonstrations
2. **Data Transforms**: Robot-specific preprocessing (`repack_transforms`, `data_transforms`)
3. **Normalization**: Z-score or quantile normalization using pre-computed statistics
4. **Model Input**: Standardized `Observation` and `Actions` structures
5. **Model Output**: Action predictions for robot execution

### Configuration System

The repository uses a hierarchical configuration system:
- `TrainConfig`: Overall training configuration
- `DataConfig`: Data processing and normalization settings
- `ModelConfig`: Model architecture parameters
- `AssetsConfig`: Asset (normalization stats) management

Configurations are defined in `src/openpi/training/config.py` with examples for different robots.

## Robot Platform Support

**ALOHA**: Bimanual manipulation tasks
- Base model: `pi0_base` (general purpose)
- Specialized: `pi0_aloha_towel`, `pi0_aloha_tupperware`, `pi0_aloha_pen_uncap`

**DROID**: Single-arm tabletop manipulation
- Model: `pi0_fast_droid` (wide range of tasks)

**Libero**: Simulation environment
- Used for fine-tuning examples and testing

## Development Notes

- **GPU Requirements**: Minimum 8GB for inference, 22.5GB for LoRA fine-tuning, 70GB for full fine-tuning
- **JAX/Flax Framework**: Uses JAX for computation, Flax NNX for neural networks
- **Remote Inference**: Supports websocket-based policy serving for off-robot inference
- **Docker Support**: Available for simplified deployment and development
- **LeRobot Integration**: Uses LeRobot datasets for training data format

## Key File Patterns

- `*_test.py`: Unit tests (run with pytest)
- `*_policy.py`: Robot-specific policy implementations
- `examples/*/`: Complete examples for different robot platforms
- `third_party/`: External robot platform code (ALOHA, Libero)
- `scripts/`: Main execution scripts for training and serving

## Testing

Run single test files:
```bash
pytest src/openpi/models/model_test.py
pytest src/openpi/policies/policy_test.py
```

Run all tests:
```bash
pytest src/ scripts/ packages/
```