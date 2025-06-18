"""Shared utilities for attention benchmarking."""

import time
import functools
from typing import Any, Callable, Dict, Tuple, Optional
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn


def create_causal_mask(seq_len: int, batch_size: int = 1) -> jnp.ndarray:
    """Create causal attention mask for decoder-style attention.
    
    Args:
        seq_len: Sequence length
        batch_size: Batch size
        
    Returns:
        Causal mask of shape (batch_size, 1, seq_len, seq_len)
    """
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    return jnp.broadcast_to(mask[None, None, :, :], (batch_size, 1, seq_len, seq_len))


def create_positions(seq_len: int, batch_size: int = 1) -> jnp.ndarray:
    """Create position indices for RoPE.
    
    Args:
        seq_len: Sequence length  
        batch_size: Batch size
        
    Returns:
        Position indices of shape (batch_size, seq_len)
    """
    positions = jnp.arange(seq_len)[None, :]
    return jnp.broadcast_to(positions, (batch_size, seq_len))


def create_random_inputs(
    batch_size: int, 
    seq_len: int, 
    width: int,
    dtype: jnp.dtype = jnp.float16,
    key: Optional[jax.random.PRNGKey] = None
) -> Dict[str, jnp.ndarray]:
    """Create random inputs for attention benchmarking.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        width: Model width/embedding dimension
        dtype: Data type for inputs
        key: JAX random key
        
    Returns:
        Dictionary with input tensors
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    
    keys = jax.random.split(key, 3)
    
    return {
        'x': jax.random.normal(keys[0], (batch_size, seq_len, width), dtype=dtype),
        'positions': create_positions(seq_len, batch_size),
        'attn_mask': create_causal_mask(seq_len, batch_size),
    }


def time_function(func: Callable, *args, warmup_runs: int = 3, benchmark_runs: int = 10, **kwargs) -> Dict[str, float]:
    """Time a function with proper JAX synchronization.
    
    Args:
        func: Function to time
        *args: Function positional arguments
        warmup_runs: Number of warmup runs for XLA compilation
        benchmark_runs: Number of benchmark runs for averaging
        **kwargs: Function keyword arguments
        
    Returns:
        Dictionary with timing statistics
    """
    # Warmup runs for XLA compilation
    for _ in range(warmup_runs):
        result = func(*args, **kwargs)
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        elif isinstance(result, (tuple, list)):
            for r in result:
                if hasattr(r, 'block_until_ready'):
                    r.block_until_ready()
    
    # Benchmark runs
    times = []
    for _ in range(benchmark_runs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        
        # Ensure computation is complete
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        elif isinstance(result, (tuple, list)):
            for r in result:
                if hasattr(r, 'block_until_ready'):
                    r.block_until_ready()
                    
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    times = np.array(times)
    return {
        'mean_time': float(np.mean(times)),
        'std_time': float(np.std(times)),
        'min_time': float(np.min(times)),
        'max_time': float(np.max(times)),
        'median_time': float(np.median(times)),
    }


def get_memory_usage() -> Dict[str, float]:
    """Get current JAX memory usage statistics.
    
    Returns:
        Dictionary with memory usage in bytes
    """
    try:
        # Get memory info from JAX backend
        backend = jax.default_backend()
        if hasattr(backend, 'get_memory_info'):
            memory_info = backend.get_memory_info()
            return {
                'total_memory': memory_info.bytes_limit,
                'used_memory': memory_info.bytes_in_use,
                'available_memory': memory_info.bytes_limit - memory_info.bytes_in_use,
            }
        else:
            # Fallback for backends without memory info
            return {
                'total_memory': 0.0,
                'used_memory': 0.0, 
                'available_memory': 0.0,
            }
    except Exception:
        # Fallback in case of any errors
        return {
            'total_memory': 0.0,
            'used_memory': 0.0,
            'available_memory': 0.0,
        }


def format_bytes(bytes_value: float) -> str:
    """Format bytes value to human readable string.
    
    Args:
        bytes_value: Value in bytes
        
    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def calculate_attention_flops(batch_size: int, seq_len: int, num_heads: int, head_dim: int) -> int:
    """Calculate theoretical FLOPs for attention computation.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        num_heads: Number of attention heads
        head_dim: Dimension per head
        
    Returns:
        Theoretical FLOP count
    """
    # QK^T: batch_size * num_heads * seq_len * seq_len * head_dim
    qk_flops = batch_size * num_heads * seq_len * seq_len * head_dim
    
    # Softmax: roughly 3 ops per element (exp, sum, div)
    softmax_flops = batch_size * num_heads * seq_len * seq_len * 3
    
    # Attention * V: batch_size * num_heads * seq_len * seq_len * head_dim  
    av_flops = batch_size * num_heads * seq_len * seq_len * head_dim
    
    return qk_flops + softmax_flops + av_flops