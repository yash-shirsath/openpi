#!/usr/bin/env python3
"""Script to inspect JAXpr before and after flash attention optimization."""

import jax
import jax.random as jrandom
from flax import nnx
from openpi.models.pi0_fast import Pi0FAST, Pi0FASTConfig
from openpi.shared import nnx_utils
from openpi.shared import array_typing as at

def inspect_model_jaxpr(model: Pi0FAST, config: Pi0FASTConfig, name="model"):
    """Inspect the JAXpr of a model's sample_actions method."""
    print(f"\n=== {name.upper()} JAXpr ===")
    
    # Create sample inputs
    rng = nnx.Rngs(0)
    observation = config.fake_obs()
    
    # Get the jitted sample_actions function
    jitted_sample_actions = nnx_utils.module_jit(model.sample_actions)
    
    # Method 1: Get JAXpr using make_jaxpr (simpler, shows logical structure)
    print("--- JAXpr (Logical Structure) ---")
    try:
        with at.disable_typechecking():
            jaxpr = jax.make_jaxpr(jitted_sample_actions)(rng, observation)
        print(jaxpr.pretty_print())
    except Exception as e:
        print(f"Error getting JAXpr: {e}")
    
    # Method 2: Get lowered HLO (shows actual compiled ops)
    print("\n--- HLO (Compiled Operations) ---")
    try:
        lowered = jitted_sample_actions.lower(rng, observation)
        hlo_text = lowered.as_text()
        # Truncate if too long
        if len(hlo_text) > 5000:
            print(hlo_text[:5000] + "\n... (truncated)")
        else:
            print(hlo_text)
    except Exception as e:
        print(f"Error getting HLO: {e}")
    
    print(f"\n=== END {name.upper()} ===")
    return jitted_sample_actions

def main():
    key = jax.random.key(0)
    config = Pi0FASTConfig()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    jitted_sample_actions = nnx_utils.module_jit(model.sample_actions)
    actions = jitted_sample_actions(key, obs)
    assert actions.shape == (batch_size, 256)

    try:
        with at.disable_typechecking():
            jaxpr = jax.make_jaxpr(jitted_sample_actions)(key, obs)
        print(jaxpr.pretty_print())
    except Exception as e:
        print(f"Error getting JAXpr: {e}")

if __name__ == "__main__":
    main() 