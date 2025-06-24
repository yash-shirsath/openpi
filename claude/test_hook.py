#!/usr/bin/env python3
"""
Simple test script to verify the attention hook functionality.
"""

import jax
import jax.numpy as jnp
import numpy as np

# Import our modified gemma_fast module
from gemma_fast_with_hooks import Attention, CAPTURE_ATTENTION_INPUTS

def test_attention_hook():
    """Test the attention hook by creating a simple attention module and running it."""
    print("Testing attention hook functionality...")
    
    # Create a simple attention module
    attention = Attention(
        num_heads=4,
        num_kv_heads=2,
        features=128,
        head_dim=32
    )
    
    # Initialize the module
    key = jax.random.PRNGKey(42)
    batch_size, seq_len = 2, 8
    
    # Create dummy inputs
    x = jax.random.normal(key, (batch_size, seq_len, 128))
    positions = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
    attn_mask = jnp.ones((batch_size, 1, seq_len, seq_len), dtype=bool)
    kv_cache = None
    decode = False
    
    # Initialize parameters
    params = attention.init(key, x, positions, attn_mask, kv_cache, decode)
    
    print(f"Hook enabled: {CAPTURE_ATTENTION_INPUTS}")
    
    if CAPTURE_ATTENTION_INPUTS:
        print("Hook is enabled - attention inputs will be saved to claude/attention_captures/")
    else:
        print("Hook is disabled - to enable, set CAPTURE_ATTENTION_INPUTS = True in gemma_fast_with_hooks.py")
    
    # Run the attention module
    try:
        output, new_kv_cache = attention.apply(params, x, positions, attn_mask, kv_cache, decode)
        print(f"✓ Attention forward pass successful!")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  KV cache initialized: {new_kv_cache is not None}")
        
        if CAPTURE_ATTENTION_INPUTS:
            print("  Check claude/attention_captures/ directory for saved inputs")
        
        return True
        
    except Exception as e:
        print(f"✗ Attention forward pass failed: {e}")
        return False

if __name__ == "__main__":
    success = test_attention_hook()
    if success:
        print("\n✓ Hook test completed successfully!")
    else:
        print("\n✗ Hook test failed!")