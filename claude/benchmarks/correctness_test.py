"""Correctness testing framework for attention implementations."""

from typing import Dict, List, Tuple, Any, Optional, Callable
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from .benchmark_config import BENCHMARK_CONFIG
from .utils import create_random_inputs, create_causal_mask, create_positions


class AttentionCorrectnessTest:
    """Correctness testing suite for attention implementations."""
    
    def __init__(self, config=BENCHMARK_CONFIG):
        self.config = config
        self.test_results = {}
        self.tolerance_rtol = 1e-4  # Relative tolerance for float comparisons
        self.tolerance_atol = 1e-6  # Absolute tolerance for float comparisons
        
    def test_attention_correctness(
        self,
        attention_module_1: nn.Module,
        attention_module_2: nn.Module,
        module_name_1: str,
        module_name_2: str,
        test_configs: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Test correctness by comparing outputs of two attention implementations.
        
        Args:
            attention_module_1: First attention module (reference)
            attention_module_2: Second attention module (test)
            module_name_1: Name of first module
            module_name_2: Name of second module
            test_configs: Optional list of test configurations
            
        Returns:
            Dictionary with correctness test results
        """
        if test_configs is None:
            test_configs = self._generate_test_configs()
        
        results = {
            'module_1': module_name_1,
            'module_2': module_name_2,
            'configuration_tests': [],
            'overall_summary': {},
            'failed_tests': [],
            'tolerance_settings': {
                'rtol': self.tolerance_rtol,
                'atol': self.tolerance_atol,
            }
        }
        
        print(f"Testing correctness: {module_name_1} vs {module_name_2}")
        
        passed_tests = 0
        total_tests = len(test_configs)
        
        for i, config in enumerate(test_configs):
            print(f"  Test {i+1}/{total_tests}: "
                  f"B={config['batch_size']}, S={config['seq_len']}, "
                  f"H={config['num_heads']}/{config['num_kv_heads']}")
            
            test_result = self._test_single_config(
                attention_module_1, attention_module_2, config
            )
            test_result.update(config)
            results['configuration_tests'].append(test_result)
            
            if test_result['passed']:
                passed_tests += 1
            else:
                results['failed_tests'].append({
                    'config_index': i,
                    'config': config,
                    'errors': test_result['errors']
                })
        
        # Overall summary
        results['overall_summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'pass_rate': passed_tests / total_tests if total_tests > 0 else 0.0,
            'all_tests_passed': passed_tests == total_tests,
        }
        
        test_key = f"{module_name_1}_vs_{module_name_2}"
        self.test_results[test_key] = results
        
        return results
    
    def test_numerical_stability(
        self,
        attention_module: nn.Module,
        module_name: str,
        test_configs: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Test numerical stability of attention implementation.
        
        Args:
            attention_module: Attention module to test
            module_name: Name of the module
            test_configs: Optional list of test configurations
            
        Returns:
            Dictionary with stability test results
        """
        if test_configs is None:
            test_configs = self._generate_test_configs()
        
        results = {
            'module_name': module_name,
            'stability_tests': [],
            'determinism_tests': [],
            'gradient_tests': [],
            'summary': {}
        }
        
        print(f"Testing numerical stability: {module_name}")
        
        for i, config in enumerate(test_configs[:5]):  # Test subset for stability
            print(f"  Stability test {i+1}/5: "
                  f"B={config['batch_size']}, S={config['seq_len']}")
            
            # Test determinism (same inputs -> same outputs)
            determinism_result = self._test_determinism(attention_module, config)
            results['determinism_tests'].append(determinism_result)
            
            # Test numerical stability with different input scales
            stability_result = self._test_input_scaling_stability(attention_module, config)
            results['stability_tests'].append(stability_result)
            
            # Test gradient computation
            gradient_result = self._test_gradient_computation(attention_module, config)
            results['gradient_tests'].append(gradient_result)
        
        # Summarize results
        results['summary'] = self._summarize_stability_tests(results)
        
        stability_key = f"{module_name}_stability"
        self.test_results[stability_key] = results
        
        return results
    
    def _generate_test_configs(self) -> List[Dict]:
        """Generate subset of test configurations for correctness testing."""
        # Use smaller subset for correctness tests to avoid excessive runtime
        configs = []
        
        # Test key configurations
        test_seq_lens = [512, 1024, 2048]  # Subset of sequence lengths
        test_batch_sizes = [1, 4]  # Subset of batch sizes
        
        for seq_len in test_seq_lens:
            for batch_size in test_batch_sizes:
                for num_heads, num_kv_heads in self.config.HEAD_CONFIGS:
                    configs.append({
                        'batch_size': batch_size,
                        'seq_len': seq_len,
                        'num_heads': num_heads,
                        'num_kv_heads': num_kv_heads,
                        'head_dim': self.config.HEAD_DIM,
                        'width': self.config.WIDTH,
                    })
        
        return configs
    
    def _test_single_config(
        self, 
        module_1: nn.Module, 
        module_2: nn.Module, 
        config: Dict
    ) -> Dict[str, Any]:
        """Test correctness for a single configuration."""
        try:
            # Create identical inputs for both modules
            key = jax.random.PRNGKey(42)  # Fixed seed for reproducibility
            inputs = create_random_inputs(
                batch_size=config['batch_size'],
                seq_len=config['seq_len'],
                width=config['width'],
                dtype=getattr(jnp, self.config.DTYPE),
                key=key
            )
            
            # Initialize both modules with same parameters
            init_key = jax.random.PRNGKey(123)
            dummy_input = inputs['x'][:1, :1]
            
            # Initialize first module
            variables_1 = module_1.init(
                init_key,
                dummy_input,
                inputs['positions'][:1, :1],
                inputs['attn_mask'][:1, :, :1, :1],
                None
            )
            module_1_bound = module_1.bind(variables_1)
            
            # Initialize second module with same parameters (if compatible)
            try:
                variables_2 = module_2.init(
                    init_key,
                    dummy_input,
                    inputs['positions'][:1, :1],
                    inputs['attn_mask'][:1, :, :1, :1],
                    None
                )
                module_2_bound = module_2.bind(variables_2)
            except Exception as e:
                # If initialization fails, modules might be incompatible
                return {
                    'passed': False,
                    'errors': [f"Module initialization failed: {str(e)}"],
                    'output_comparison': {},
                    'kv_cache_comparison': {},
                }
            
            # Run forward pass on both modules
            output_1, kv_cache_1 = module_1_bound(
                inputs['x'],
                inputs['positions'],
                inputs['attn_mask'],
                None
            )
            
            output_2, kv_cache_2 = module_2_bound(
                inputs['x'],
                inputs['positions'],
                inputs['attn_mask'],
                None
            )
            
            # Compare outputs
            output_comparison = self._compare_outputs(output_1, output_2)
            kv_cache_comparison = self._compare_kv_caches(kv_cache_1, kv_cache_2)
            
            # Check if all comparisons passed
            all_passed = (
                output_comparison['outputs_match'] and
                kv_cache_comparison['caches_match']
            )
            
            errors = []
            if not output_comparison['outputs_match']:
                errors.extend(output_comparison['errors'])
            if not kv_cache_comparison['caches_match']:
                errors.extend(kv_cache_comparison['errors'])
            
            return {
                'passed': all_passed,
                'errors': errors,
                'output_comparison': output_comparison,
                'kv_cache_comparison': kv_cache_comparison,
            }
            
        except Exception as e:
            return {
                'passed': False,
                'errors': [f"Test execution failed: {str(e)}"],
                'output_comparison': {},
                'kv_cache_comparison': {},
            }
    
    def _compare_outputs(self, output_1, output_2) -> Dict[str, Any]:
        """Compare attention outputs between two implementations."""
        try:
            # Handle different output formats
            if isinstance(output_1, (list, tuple)) and isinstance(output_2, (list, tuple)):
                # Multi-output case (e.g., multiple experts)
                if len(output_1) != len(output_2):
                    return {
                        'outputs_match': False,
                        'errors': [f"Output length mismatch: {len(output_1)} vs {len(output_2)}"],
                        'max_diff': float('inf'),
                        'relative_error': float('inf'),
                    }
                
                max_diffs = []
                relative_errors = []
                
                for i, (out_1, out_2) in enumerate(zip(output_1, output_2)):
                    if out_1 is None and out_2 is None:
                        continue
                    if out_1 is None or out_2 is None:
                        return {
                            'outputs_match': False,
                            'errors': [f"Output {i} None mismatch: {out_1 is None} vs {out_2 is None}"],
                            'max_diff': float('inf'),
                            'relative_error': float('inf'),
                        }
                    
                    diff = jnp.abs(out_1 - out_2)
                    max_diff = float(jnp.max(diff))
                    relative_error = float(jnp.max(diff / (jnp.abs(out_1) + 1e-8)))
                    
                    max_diffs.append(max_diff)
                    relative_errors.append(relative_error)
                
                overall_max_diff = max(max_diffs) if max_diffs else 0.0
                overall_relative_error = max(relative_errors) if relative_errors else 0.0
                
            else:
                # Single output case
                diff = jnp.abs(output_1 - output_2)
                overall_max_diff = float(jnp.max(diff))
                overall_relative_error = float(jnp.max(diff / (jnp.abs(output_1) + 1e-8)))
            
            # Check if outputs match within tolerance
            outputs_match = (
                overall_max_diff < self.tolerance_atol or
                overall_relative_error < self.tolerance_rtol
            )
            
            errors = []
            if not outputs_match:
                errors.append(f"Output mismatch: max_diff={overall_max_diff:.2e}, rel_error={overall_relative_error:.2e}")
            
            return {
                'outputs_match': outputs_match,
                'errors': errors,
                'max_diff': overall_max_diff,
                'relative_error': overall_relative_error,
            }
            
        except Exception as e:
            return {
                'outputs_match': False,
                'errors': [f"Output comparison failed: {str(e)}"],
                'max_diff': float('inf'),
                'relative_error': float('inf'),
            }
    
    def _compare_kv_caches(self, cache_1, cache_2) -> Dict[str, Any]:
        """Compare KV caches between two implementations."""
        try:
            if cache_1 is None and cache_2 is None:
                return {
                    'caches_match': True,
                    'errors': [],
                    'k_max_diff': 0.0,
                    'v_max_diff': 0.0,
                }
            
            if cache_1 is None or cache_2 is None:
                return {
                    'caches_match': False,
                    'errors': [f"Cache None mismatch: {cache_1 is None} vs {cache_2 is None}"],
                    'k_max_diff': float('inf'),
                    'v_max_diff': float('inf'),
                }
            
            # Extract K and V from caches
            k_1, v_1 = cache_1
            k_2, v_2 = cache_2
            
            # Compare K and V separately
            k_diff = jnp.abs(k_1 - k_2)
            v_diff = jnp.abs(v_1 - v_2)
            
            k_max_diff = float(jnp.max(k_diff))
            v_max_diff = float(jnp.max(v_diff))
            
            k_rel_error = float(jnp.max(k_diff / (jnp.abs(k_1) + 1e-8)))
            v_rel_error = float(jnp.max(v_diff / (jnp.abs(v_1) + 1e-8)))
            
            k_match = k_max_diff < self.tolerance_atol or k_rel_error < self.tolerance_rtol
            v_match = v_max_diff < self.tolerance_atol or v_rel_error < self.tolerance_rtol
            
            caches_match = k_match and v_match
            
            errors = []
            if not k_match:
                errors.append(f"K cache mismatch: max_diff={k_max_diff:.2e}, rel_error={k_rel_error:.2e}")
            if not v_match:
                errors.append(f"V cache mismatch: max_diff={v_max_diff:.2e}, rel_error={v_rel_error:.2e}")
            
            return {
                'caches_match': caches_match,
                'errors': errors,
                'k_max_diff': k_max_diff,
                'v_max_diff': v_max_diff,
                'k_rel_error': k_rel_error,
                'v_rel_error': v_rel_error,
            }
            
        except Exception as e:
            return {
                'caches_match': False,
                'errors': [f"Cache comparison failed: {str(e)}"],
                'k_max_diff': float('inf'),
                'v_max_diff': float('inf'),
            }
    
    def _test_determinism(self, attention_module: nn.Module, config: Dict) -> Dict[str, Any]:
        """Test that attention module produces deterministic outputs."""
        key = jax.random.PRNGKey(42)
        inputs = create_random_inputs(
            batch_size=config['batch_size'],
            seq_len=config['seq_len'],
            width=config['width'],
            dtype=getattr(jnp, self.config.DTYPE),
            key=key
        )
        
        # Initialize module
        init_key = jax.random.PRNGKey(123)
        dummy_input = inputs['x'][:1, :1]
        variables = attention_module.init(
            init_key,
            dummy_input,
            inputs['positions'][:1, :1],
            inputs['attn_mask'][:1, :, :1, :1],
            None
        )
        bound_module = attention_module.bind(variables)
        
        # Run multiple times with same inputs
        outputs = []
        for _ in range(3):
            output, _ = bound_module(
                inputs['x'],
                inputs['positions'],
                inputs['attn_mask'],
                None
            )
            outputs.append(output)
        
        # Compare outputs for determinism
        max_diff = 0.0
        for i in range(1, len(outputs)):
            if isinstance(outputs[0], (list, tuple)):
                for out_0, out_i in zip(outputs[0], outputs[i]):
                    if out_0 is not None and out_i is not None:
                        diff = float(jnp.max(jnp.abs(out_0 - out_i)))
                        max_diff = max(max_diff, diff)
            else:
                diff = float(jnp.max(jnp.abs(outputs[0] - outputs[i])))
                max_diff = max(max_diff, diff)
        
        is_deterministic = max_diff < 1e-12  # Very strict tolerance for determinism
        
        return {
            'is_deterministic': is_deterministic,
            'max_difference': max_diff,
            'config': config,
        }
    
    def _test_input_scaling_stability(self, attention_module: nn.Module, config: Dict) -> Dict[str, Any]:
        """Test numerical stability with different input scales."""
        scales = [0.1, 1.0, 10.0]  # Different input scales
        stability_results = []
        
        for scale in scales:
            key = jax.random.PRNGKey(42)
            inputs = create_random_inputs(
                batch_size=config['batch_size'],
                seq_len=config['seq_len'],
                width=config['width'],
                dtype=getattr(jnp, self.config.DTYPE),
                key=key
            )
            
            # Scale the inputs
            inputs['x'] = inputs['x'] * scale
            
            # Initialize and run module
            try:
                init_key = jax.random.PRNGKey(123)
                dummy_input = inputs['x'][:1, :1]
                variables = attention_module.init(
                    init_key,
                    dummy_input,
                    inputs['positions'][:1, :1],
                    inputs['attn_mask'][:1, :, :1, :1],
                    None
                )
                bound_module = attention_module.bind(variables)
                
                output, _ = bound_module(
                    inputs['x'],
                    inputs['positions'],
                    inputs['attn_mask'],
                    None
                )
                
                # Check for NaN or Inf
                has_nan = False
                has_inf = False
                
                if isinstance(output, (list, tuple)):
                    for out in output:
                        if out is not None:
                            has_nan = has_nan or bool(jnp.any(jnp.isnan(out)))
                            has_inf = has_inf or bool(jnp.any(jnp.isinf(out)))
                else:
                    has_nan = bool(jnp.any(jnp.isnan(output)))
                    has_inf = bool(jnp.any(jnp.isinf(output)))
                
                stability_results.append({
                    'scale': scale,
                    'has_nan': has_nan,
                    'has_inf': has_inf,
                    'stable': not (has_nan or has_inf),
                })
                
            except Exception as e:
                stability_results.append({
                    'scale': scale,
                    'has_nan': True,
                    'has_inf': True,
                    'stable': False,
                    'error': str(e),
                })
        
        all_stable = all(result['stable'] for result in stability_results)
        
        return {
            'all_scales_stable': all_stable,
            'scale_results': stability_results,
            'config': config,
        }
    
    def _test_gradient_computation(self, attention_module: nn.Module, config: Dict) -> Dict[str, Any]:
        """Test that gradients can be computed successfully."""
        try:
            key = jax.random.PRNGKey(42)
            inputs = create_random_inputs(
                batch_size=config['batch_size'],
                seq_len=config['seq_len'],
                width=config['width'],
                dtype=getattr(jnp, self.config.DTYPE),
                key=key
            )
            
            # Define loss function
            def loss_fn(params):
                bound_module = attention_module.bind({'params': params})
                output, _ = bound_module(
                    inputs['x'],
                    inputs['positions'],
                    inputs['attn_mask'],
                    None
                )
                
                # Simple loss: sum of squared outputs
                if isinstance(output, (list, tuple)):
                    loss = sum(jnp.sum(out**2) for out in output if out is not None)
                else:
                    loss = jnp.sum(output**2)
                
                return loss
            
            # Initialize module
            init_key = jax.random.PRNGKey(123)
            dummy_input = inputs['x'][:1, :1]
            variables = attention_module.init(
                init_key,
                dummy_input,
                inputs['positions'][:1, :1],
                inputs['attn_mask'][:1, :, :1, :1],
                None
            )
            
            # Compute gradients
            loss_value, grads = jax.value_and_grad(loss_fn)(variables['params'])
            
            # Check for NaN/Inf in gradients
            def check_grads(grad_tree):
                has_nan = False
                has_inf = False
                
                def check_leaf(x):
                    nonlocal has_nan, has_inf
                    if isinstance(x, jnp.ndarray):
                        has_nan = has_nan or bool(jnp.any(jnp.isnan(x)))
                        has_inf = has_inf or bool(jnp.any(jnp.isinf(x)))
                
                jax.tree_map(check_leaf, grad_tree)
                return has_nan, has_inf
            
            grad_has_nan, grad_has_inf = check_grads(grads)
            
            return {
                'gradients_computed': True,
                'loss_value': float(loss_value),
                'grad_has_nan': grad_has_nan,
                'grad_has_inf': grad_has_inf,
                'gradients_valid': not (grad_has_nan or grad_has_inf),
                'config': config,
            }
            
        except Exception as e:
            return {
                'gradients_computed': False,
                'error': str(e),
                'config': config,
            }
    
    def _summarize_stability_tests(self, results: Dict) -> Dict[str, Any]:
        """Summarize stability test results."""
        determinism_tests = results['determinism_tests']
        stability_tests = results['stability_tests']
        gradient_tests = results['gradient_tests']
        
        # Determinism summary
        deterministic_count = sum(1 for test in determinism_tests if test['is_deterministic'])
        determinism_rate = deterministic_count / len(determinism_tests) if determinism_tests else 0.0
        
        # Stability summary
        stable_count = sum(1 for test in stability_tests if test['all_scales_stable'])
        stability_rate = stable_count / len(stability_tests) if stability_tests else 0.0
        
        # Gradient summary
        grad_valid_count = sum(1 for test in gradient_tests 
                              if test.get('gradients_computed', False) and test.get('gradients_valid', False))
        gradient_rate = grad_valid_count / len(gradient_tests) if gradient_tests else 0.0
        
        return {
            'determinism': {
                'pass_rate': determinism_rate,
                'passed_tests': deterministic_count,
                'total_tests': len(determinism_tests),
            },
            'numerical_stability': {
                'pass_rate': stability_rate,
                'passed_tests': stable_count,
                'total_tests': len(stability_tests),
            },
            'gradient_computation': {
                'pass_rate': gradient_rate,
                'passed_tests': grad_valid_count,
                'total_tests': len(gradient_tests),
            },
            'overall_stability': {
                'all_tests_passed': (determinism_rate == 1.0 and 
                                    stability_rate == 1.0 and 
                                    gradient_rate == 1.0),
                'average_pass_rate': (determinism_rate + stability_rate + gradient_rate) / 3,
            }
        }
    
    def print_correctness_results(self, test_key: str):
        """Print formatted correctness test results."""
        if test_key not in self.test_results:
            print(f"No correctness results found for: {test_key}")
            return
        
        results = self.test_results[test_key]
        
        if 'overall_summary' in results:
            # Comparison test results
            summary = results['overall_summary']
            
            print(f"\n{'='*60}")
            print(f"Correctness Test Results: {results['module_1']} vs {results['module_2']}")
            print(f"{'='*60}")
            
            print(f"\nOverall Summary:")
            print(f"  Total tests: {summary['total_tests']}")
            print(f"  Passed tests: {summary['passed_tests']}")
            print(f"  Failed tests: {summary['failed_tests']}")
            print(f"  Pass rate: {summary['pass_rate']:.1%}")
            print(f"  All tests passed: {summary['all_tests_passed']}")
            
            if results['failed_tests']:
                print(f"\nFailed Test Details:")
                for failure in results['failed_tests'][:3]:  # Show first 3 failures
                    config = failure['config']
                    print(f"  Config: B={config['batch_size']}, S={config['seq_len']}, H={config['num_heads']}")
                    for error in failure['errors']:
                        print(f"    - {error}")
        
        else:
            # Stability test results
            summary = results['summary']
            
            print(f"\n{'='*60}")
            print(f"Stability Test Results: {results['module_name']}")
            print(f"{'='*60}")
            
            print(f"\nDeterminism Tests:")
            det = summary['determinism']
            print(f"  Pass rate: {det['pass_rate']:.1%} ({det['passed_tests']}/{det['total_tests']})")
            
            print(f"\nNumerical Stability Tests:")
            stab = summary['numerical_stability']
            print(f"  Pass rate: {stab['pass_rate']:.1%} ({stab['passed_tests']}/{stab['total_tests']})")
            
            print(f"\nGradient Computation Tests:")
            grad = summary['gradient_computation']
            print(f"  Pass rate: {grad['pass_rate']:.1%} ({grad['passed_tests']}/{grad['total_tests']})")
            
            print(f"\nOverall Stability:")
            overall = summary['overall_stability']
            print(f"  All tests passed: {overall['all_tests_passed']}")
            print(f"  Average pass rate: {overall['average_pass_rate']:.1%}")
    
    def save_correctness_results(self, filepath: str):
        """Save correctness test results to file."""
        import json
        
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(self.test_results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Correctness results saved to: {filepath}")