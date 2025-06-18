# Implementation Plan: Integrating Flash Attention into Gemma Models

This document outlines the plan to integrate Flash Attention into the `gemma` and `gemma_fast` models within this repository to improve performance and reduce memory usage.

## 1. Objective

The primary goal is to replace the standard dot-product attention mechanism implemented with `jnp.einsum` in `src/openpi/models/gemma.py` and `src/openpi/models/gemma_fast.py` with a more efficient Flash Attention implementation. This change is expected to yield significant speedups and memory savings, especially for long sequences.

## 2. Background

### Current Implementation

Both `gemma.py` and `gemma_fast.py` use a standard self-attention mechanism. The core logic in the `Attention` module of both files is approximately:

```python
logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k)
masked_logits = jnp.where(attn_mask, logits, big_neg)
probs = jax.nn.softmax(masked_logits, axis=-1)
encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
```

This is a memory-intensive operation because it materializes the large `(T, S)` attention matrix.

### Flash Attention

Flash Attention is a fast and memory-efficient exact attention algorithm that avoids materializing the full attention matrix. It computes the attention output in blocks, leveraging hardware-specific optimizations.

### Recommended Library: `Kvax`

After reviewing several options, `Kvax` is recommended. It has good support for advanced masking, which is a key requirement for this codebase.

- **URL**: [https://nebius.com/blog/posts/kvax-open-source-flash-attention-for-jax](https://nebius.com/blog/posts/kvax-open-source-flash-attention-for-jax)
- **Alternative**: `nshepperd/flash_attn_jax` ([https://github.com/nshepperd/flash_attn_jax](https://github.com/nshepperd/flash_attn_jax)) is another option. It provides JAX bindings for the official Flash Attention 2 CUDA kernels, which should deliver high performance, but its masking support is more limited.

**Important consideration:** The current attention implementation uses a generic `attn_mask`. `Kvax` is better suited to handle the padding masks used in this codebase, which is why it is the primary recommendation. `nshepperd/flash_attn_jax` primarily uses a simple `is_causal` flag and may not be sufficient.

## 3. Benchmarking Plan

To verify the performance improvements, a robust benchmarking strategy is required.

1.  **Create a Benchmarking Script**: A new script, e.g., `scripts/benchmark_attention.py`, should be created.
2.  **Metrics**:
    - **Speed**: Measure training steps per second and inference tokens per second.
    - **Memory**: Profile GPU memory usage (peak and average) during training and inference. JAX's `device_memory_profile` can be used.
3.  **Scenarios**:
    - **Varying Sequence Length**: Benchmark with different sequence lengths (e.g., 512, 1024, 2048, 4096) to see how performance scales.
    - **Training vs. Inference**: Test both modes. For `gemma_fast`, this means benchmarking the autoregressive decoding with the KV cache.
4.  **Procedure**:
    - Run the benchmark on the original implementation to establish a baseline.
    - Run the same benchmark on the Flash Attention implementation.
    - Compare results for `gemma` and `gemma_fast` separately.

## 4. Testing Plan

Ensuring the correctness of the new implementation is critical.

1.  **Numerical Equivalence**:

    - Create a test to compare the output of the Flash Attention implementation with the original `einsum`-based attention.
    - Given the same inputs, the outputs should be numerically close. Use `jnp.allclose` with appropriate tolerances (`atol`, `rtol`).
    - This test should be run with `deterministic=True` (no dropout) to ensure comparability.
    - Test both forward and backward passes (gradients).

2.  **Integration Tests**:

    - Run all existing model and training tests (`*_test.py`) with Flash Attention enabled. This ensures that the new implementation does not break any downstream components.

3.  **End-to-End Validation**:
    - Perform a full training run of a model (e.g., using `scripts/train.py`) with Flash Attention enabled.
    - Monitor the loss curve and final model performance to ensure that model convergence and final accuracy are not negatively affected.

## 5. Implementation Steps

### Step 1: Add Dependency

Add `kvax` to the project's dependencies. This will likely be in `pyproject.toml` or a `requirements.txt` file.

```bash
# Example of adding with pip/uv
pip install kvax
```

### Step 2: Modify `Attention` Modules

The `__call__` method of the `Attention` class in both `src/openpi/models/gemma.py` and `src/openpi/models/gemma_fast.py` needs to be modified.

1.  **Import `flash_mha` from `kvax`**:

    ```python
    # Note: The exact import path may differ.
    from kvax.flash import flash_mha
    ```

2.  **Replace `einsum`-based attention**: The section calculating `logits`, `probs`, and `encoded` should be replaced with a single call to `flash_mha`.

    The current shapes are:

    - `q`: `(B, T, N, H)` (Batch, Query SeqLen, Num Query Heads, Head Dim)
    - `k`: `(B, S, K, H)` (Batch, KV SeqLen, Num KV Heads, Head Dim)
    - `v`: `(B, S, K, H)`

    The `flash_mha` function from `kvax` should support these shapes and custom masking.

    **Proposed Change:**

    ```python
    # In src/openpi/models/gemma.py and src/openpi/models/gemma_fast.py
    # inside Attention.__call__

    # ... after q, k, v are computed and ROPE is applied ...

    # The original einsum implementation will be replaced.
    # The 'attn_mask' is of shape (B, 1, T, S). Kvax is expected
    # to handle this mask directly.

    # Rearrange q, k, v to be compatible with flash_mha if necessary.
    # flash_mha expects (batch, seq_len, num_heads, head_dim)
    # q: (B, T, N, H)
    # k: (B, S, K, H)
    # v: (B, S, K, H)
    # These seem compatible.

    # Gemma normalizes q by head_dim**-0.5. flash_mha does this by default.
    # We should pass softmax_scale=1.0 if q is already scaled.
    # q *= self.configs[0].head_dim ** -0.5

    # The output of flash_mha will have the same shape as q.
    # The rearrange from (B, T, K, G, H) to (B, T, N, H) is no longer needed before the output projection.
    encoded = flash_mha(q, k, v, mask=attn_mask) # Verify exact parameter name for mask

    # ... rest of the function (output projection) ...
    ```

3.  **Configuration Flag**: Introduce a configuration option (e.g., `use_flash_attention: bool`) in the model `Config` to allow switching between the original attention and Flash Attention. This will be invaluable for debugging and benchmarking.

    ```python
    # in Config class
    use_flash_attention: bool = False

    # in Attention.__call__
    if self.config.use_flash_attention:
        # ... flash attention implementation ...
    else:
        # ... original einsum implementation ...
    ```

## 6. Rollout Plan

1.  Implement the changes on a separate branch.
2.  Initially, the `use_flash_attention` flag will be `False` by default.
3.  After all tests and benchmarks pass and show positive results, a PR can be opened to merge the changes.
4.  The default can be switched to `True` in a follow-up, or it can be left as a user-configurable option for maximum flexibility.
