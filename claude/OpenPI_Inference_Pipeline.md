# OpenPI Inference Pipeline

This document explains the OpenPI inference pipeline architecture, models involved, and data flow.

## Main Inference Pipeline Components

### 1. Policy Layer
- **[`Policy` class](src/openpi/policies/policy.py)**: Main inference orchestrator that wraps model execution
- **Key method**: `infer(obs: dict) -> dict` - the main inference entry point at [policy.py:85](src/openpi/policies/policy.py#L85)
- Applies input transforms → model inference → output transforms
- Handles batching/unbatching and timing measurements

### 2. Policy Server
- **[`WebsocketPolicyServer`](src/openpi/serving/websocket_policy_server.py)**: Serves policies over WebSocket protocol
- Handles client connections and real-time inference requests at [websocket_policy_server.py:45](src/openpi/serving/websocket_policy_server.py#L45)
- Provides health checks and error handling

### 3. Policy Configuration Factory
- **[`create_trained_policy()`](src/openpi/policies/policy_config.py)**: Creates trained policies from checkpoints
- Loads model weights, normalization stats, and sets up transform pipelines at [policy_config.py:156](src/openpi/policies/policy_config.py#L156)

## Models Involved and Their Roles

### Primary Model Architectures

#### 1. Pi0 Model
**File**: [`src/openpi/models/pi0.py`](src/openpi/models/pi0.py)

- **Type**: Diffusion-based transformer policy
- **Components**:
  - **PaliGemma backbone**: Combines vision (SigLIP) + language (Gemma) processing
  - **Vision encoder**: SigLIP-So400m/14 for image understanding
  - **Language model**: Gemma (300M or 2B variants) for instruction processing
  - **Action expert**: Separate Gemma model (300M) for action prediction
- **Process**: Uses diffusion sampling with iterative denoising for action generation
- **Input**: Multi-view images + robot state + text prompts
- **Output**: Action sequences (typically 50 timesteps)

#### 2. Pi0-FAST Model
**File**: [`src/openpi/models/pi0_fast.py`](src/openpi/models/pi0_fast.py)

- **Type**: Autoregressive transformer policy (faster inference)
- **Components**:
  - **PaliGemma backbone**: Same vision+language processing as Pi0
  - **FAST tokenizer**: Discretizes actions into tokens for autoregressive generation
- **Process**: Generates actions token-by-token autoregressively
- **Input**: Multi-view images + robot state + text prompts  
- **Output**: Tokenized action sequences (converted back to continuous actions)

### Sub-Models and Components

#### 3. SigLIP Vision Encoder
**File**: [`src/openpi/models/siglip.py`](src/openpi/models/siglip.py)

- **Role**: Processes multi-view camera images (typically 3 views: base, left wrist, right wrist)
- **Architecture**: Vision Transformer with 224x224 input resolution
- **Variant**: So400m/14 (400M parameters, 14x14 patch size)

#### 4. Gemma Language Models
**File**: [`src/openpi/models/gemma.py`](src/openpi/models/gemma.py)

- **Variants Available**:
  - `gemma_300m`: 311M parameters, 18 layers, 1024 width
  - `gemma_2b`: 2B parameters, 18 layers, 2048 width
  - `*_lora` variants: LoRA fine-tuning versions for memory efficiency
- **Role**: Processes text instructions and integrates with vision features

#### 5. Tokenizers
**File**: [`src/openpi/models/tokenizer.py`](src/openpi/models/tokenizer.py)

- **PaliGemma Tokenizer**: For Pi0 model text processing
- **FAST Tokenizer**: For Pi0-FAST model, handles both text and action tokenization

## Data Flow Through the Pipeline

### 1. Input Processing
```
Raw Observations → Robot-specific transforms → Normalization → Model-specific transforms
```

### 2. Model Inference
```
Images → SigLIP encoder → Vision tokens
Text → Gemma tokenizer → Language tokens
State → Projection layers → State tokens

Combined tokens → Transformer → Action predictions
```

### 3. Output Processing
```
Model outputs → Unnormalization → Robot-specific output transforms → Actions
```

## Key Configuration Files and Available Models

### Pre-trained Checkpoints
**File**: [`scripts/serve_policy.py`](scripts/serve_policy.py)

- **`pi0_base`**: Base Pi0 model for ALOHA robots
- **`pi0_aloha_sim`**: Pi0 trained on ALOHA simulation data
- **`pi0_fast_droid`**: Pi0-FAST model for DROID robots
- **`pi0_fast_libero`**: Pi0-FAST model for Libero tasks

### Training Configurations
**File**: [`src/openpi/training/config.py`](src/openpi/training/config.py)

- **Inference configs**: `pi0_aloha`, `pi0_droid`, `pi0_fast_droid`
- **Fine-tuning configs**: `pi0_libero`, `pi0_fast_libero`, with LoRA variants
- **Custom dataset configs**: Templates for ALOHA, DROID, and Libero environments

## Robot Platform Support

1. **ALOHA**: Bimanual manipulation with 4 camera views
2. **DROID**: Single-arm manipulation with external + wrist cameras
3. **Libero**: Simulated manipulation tasks
4. **Custom platforms**: Extensible through transform functions

## Key Files That Orchestrate the Pipeline

### Main Entry Points
- [`scripts/serve_policy.py`](scripts/serve_policy.py) - Policy serving
- [`examples/inference.ipynb`](examples/inference.ipynb) - Direct inference examples

### Core Orchestration
- [`src/openpi/policies/policy.py`](src/openpi/policies/policy.py) - Main Policy class
- [`src/openpi/policies/policy_config.py`](src/openpi/policies/policy_config.py) - Policy factory
- [`src/openpi/models/model.py`](src/openpi/models/model.py) - Base model interface

### Transform Pipeline
- [`src/openpi/transforms.py`](src/openpi/transforms.py) - Data transformation framework
- Robot-specific policies (e.g., [`droid_policy.py`](src/openpi/policies/droid_policy.py)) for input/output adaptation

## Architecture Summary

The OpenPI architecture is designed for modularity, allowing different robot platforms to plug in through transform functions while sharing the same underlying vision-language-action models. The pipeline supports both diffusion-based (Pi0) and autoregressive (Pi0-FAST) inference modes, with the latter optimized for faster real-time performance.