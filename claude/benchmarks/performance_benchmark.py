"""Performance benchmarking for attention implementations."""

import time
from typing import Dict, List, Tuple, Any, Optional
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from .benchmark_config import BENCHMARK_CONFIG
from .utils import create_random_inputs, time_function, calculate_attention_flops


class AttentionPerformanceBenchmark:
    """Performance benchmarking suite for attention implementations."""
    
    def __init__(self, config=BENCHMARK_CONFIG):
        self.config = config
        self.results = {}
        
    def benchmark_attention_module(
        self, 
        attention_module: nn.Module,
        module_name: str,
        test_configs: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Benchmark an attention module across different configurations.
        
        Args:
            attention_module: The attention module to benchmark
            module_name: Name identifier for the module
            test_configs: Optional list of test configurations
            
        Returns:
            Dictionary with benchmark results
        """
        if test_configs is None:
            # Infer architecture from the module to generate correct configs
            try:
                num_heads = attention_module.num_heads
                num_kv_heads = attention_module.num_kv_heads
            except AttributeError:
                print(f"Warning: Could not determine architecture from module {module_name}. "
                      "Using default head configs from benchmark_config. This may be incorrect.")
                num_heads = None
                num_kv_heads = None

            test_configs = self._generate_test_configs(
                num_heads=num_heads,
                num_kv_heads=num_kv_heads
            )
            
        results = {
            'module_name': module_name,
            'configurations': [],
            'summary_stats': {}
        }
        
        print(f"Benchmarking {module_name}...")
        
        for i, config in enumerate(test_configs):
            print(f"  Config {i+1}/{len(test_configs)}: "
                  f"B={config['batch_size']}, S={config['seq_len']}, "
                  f"H={config['num_heads']}/{config['num_kv_heads']}")
            
            config_result = self._benchmark_single_config(attention_module, config)
            config_result.update(config)
            results['configurations'].append(config_result)
            
        # Compute summary statistics
        results['summary_stats'] = self._compute_summary_stats(results['configurations'])
        
        self.results[module_name] = results
        return results
    
    def _generate_test_configs(
        self, 
        num_heads: Optional[int] = None, 
        num_kv_heads: Optional[int] = None
    ) -> List[Dict]:
        """Generate test configurations from benchmark config."""
        configs = []
        
        head_configs = self.config.HEAD_CONFIGS
        if num_heads is not None and num_kv_heads is not None:
            head_configs = [(num_heads, num_kv_heads)]

        for seq_len in self.config.SEQUENCE_LENGTHS:
            for batch_size in self.config.BATCH_SIZES:
                for h, kv_h in head_configs:
                    configs.append({
                        'batch_size': batch_size,
                        'seq_len': seq_len,
                        'num_heads': h,
                        'num_kv_heads': kv_h,
                        'head_dim': self.config.HEAD_DIM,
                        'width': self.config.WIDTH,
                    })
        
        return configs
    
    def _benchmark_single_config(self, attention_module: nn.Module, config: Dict) -> Dict[str, Any]:
        """Benchmark attention module for a single configuration."""
        # Create inputs
        inputs = create_random_inputs(
            batch_size=config['batch_size'],
            seq_len=config['seq_len'], 
            width=config['width'],
            dtype=getattr(jnp, self.config.DTYPE)
        )
        
        # Initialize module if needed
        if not hasattr(attention_module, 'params') or attention_module.params is None:
            key = jax.random.PRNGKey(42)
            dummy_input = inputs['x'][:1, :1]  # Small input for initialization
            variables = attention_module.init(
                key, 
                dummy_input, 
                inputs['positions'][:1, :1],
                inputs['attn_mask'][:1, :, :1, :1],
                None  # kv_cache
            )
            attention_module = attention_module.bind(variables)
        
        # Create forward function
        def forward_fn():
            return attention_module(
                inputs['x'],
                inputs['positions'], 
                inputs['attn_mask'],
                None  # kv_cache for now
            )
        
        # Time the forward pass
        timing_stats = time_function(
            forward_fn,
            warmup_runs=self.config.WARMUP_RUNS,
            benchmark_runs=self.config.BENCHMARK_RUNS
        )
        
        # Calculate derived metrics
        theoretical_flops = calculate_attention_flops(
            config['batch_size'], config['seq_len'], 
            config['num_heads'], config['head_dim']
        )
        
        tokens_per_second = (config['batch_size'] * config['seq_len']) / timing_stats['mean_time']
        flops_per_second = theoretical_flops / timing_stats['mean_time']
        
        return {
            **timing_stats,
            'theoretical_flops': theoretical_flops,
            'tokens_per_second': tokens_per_second,
            'flops_per_second': flops_per_second,
            'memory_usage': self._measure_memory_usage(forward_fn),
        }
    
    def _measure_memory_usage(self, forward_fn) -> Dict[str, float]:
        """Measure memory usage during forward pass."""
        # This function is not perfectly accurate but gives a reasonable estimate.
        # For GPU, it measures the increase in bytes allocated on the device.
        devices = jax.local_devices()
        if not devices:
            initial_memory = 0
        else:
            # Assuming we're benchmarking on the first device
            try:
                devices[0].synchronize()
                initial_memory = devices[0].get_memory_info().bytes_in_use
            except Exception:
                initial_memory = 0 # Fallback for different JAX versions or backends

        # Run forward pass
        result = forward_fn()
        
        # Ensure computation is finished before measuring memory
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        elif isinstance(result, (tuple, list)):
            for r in result:
                if hasattr(r, 'block_until_ready'):
                    r.block_until_ready()
        
        if not devices:
            memory_delta = 0
        else:
            try:
                devices[0].synchronize()
                peak_memory = devices[0].get_memory_info().bytes_in_use
                memory_delta = peak_memory - initial_memory
            except Exception:
                memory_delta = 0 # Fallback
            
        return {
            'memory_delta_bytes': memory_delta,
            'memory_delta_mb': memory_delta / (1024 * 1024),
        }
    
    def _compute_summary_stats(self, configurations: List[Dict]) -> Dict[str, Any]:
        """Compute summary statistics across all configurations."""
        times = [config['mean_time'] for config in configurations]
        tokens_per_sec = [config['tokens_per_second'] for config in configurations]  
        memory_usage = [config['memory_usage']['memory_delta_mb'] for config in configurations]
        
        return {
            'total_configs': len(configurations),
            'time_stats': {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times), 
                'max': np.max(times),
            },
            'throughput_stats': {
                'mean_tokens_per_sec': np.mean(tokens_per_sec),
                'max_tokens_per_sec': np.max(tokens_per_sec),
                'min_tokens_per_sec': np.min(tokens_per_sec),
            },
            'memory_stats': {
                'mean_memory_mb': np.mean(memory_usage),
                'max_memory_mb': np.max(memory_usage),
                'min_memory_mb': np.min(memory_usage),
            }
        }
    
    def compare_modules(self, module_names: List[str]) -> Dict[str, Any]:
        """Compare performance between multiple benchmarked modules."""
        if not all(name in self.results for name in module_names):
            missing = [name for name in module_names if name not in self.results]
            raise ValueError(f"Missing benchmark results for modules: {missing}")
        
        comparison = {
            'modules': module_names,
            'relative_performance': {},
            'scaling_analysis': {}
        }
        
        # Use first module as baseline
        baseline_name = module_names[0]
        baseline_stats = self.results[baseline_name]['summary_stats']
        
        for module_name in module_names[1:]:
            module_stats = self.results[module_name]['summary_stats']
            
            # Calculate relative performance metrics with safe division
            baseline_memory = baseline_stats['memory_stats']['mean_memory_mb']
            module_memory = module_stats['memory_stats']['mean_memory_mb']
            
            comparison['relative_performance'][module_name] = {
                'time_ratio': module_stats['time_stats']['mean'] / baseline_stats['time_stats']['mean'],
                'throughput_ratio': module_stats['throughput_stats']['mean_tokens_per_sec'] / baseline_stats['throughput_stats']['mean_tokens_per_sec'],
                'memory_ratio': module_memory / baseline_memory if baseline_memory != 0 else float('inf') if module_memory != 0 else 1.0,
            }
        
        # Analyze scaling behavior
        for module_name in module_names:
            scaling_analysis = self._analyze_scaling(self.results[module_name]['configurations'])
            comparison['scaling_analysis'][module_name] = scaling_analysis
            
        return comparison
    
    def _analyze_scaling(self, configurations: List[Dict]) -> Dict[str, Any]:
        """Analyze how performance scales with sequence length and batch size."""
        # Group by sequence length
        seq_len_groups = {}
        for config in configurations:
            seq_len = config['seq_len']
            if seq_len not in seq_len_groups:
                seq_len_groups[seq_len] = []
            seq_len_groups[seq_len].append(config)
        
        # Analyze sequence length scaling
        seq_lens = sorted(seq_len_groups.keys())
        seq_len_times = []
        seq_len_memory = []
        
        for seq_len in seq_lens:
            configs = seq_len_groups[seq_len]
            avg_time = np.mean([c['mean_time'] for c in configs])
            avg_memory = np.mean([c['memory_usage']['memory_delta_mb'] for c in configs])
            seq_len_times.append(avg_time)
            seq_len_memory.append(avg_memory)
        
        # Fit scaling trends (linear in log space for power law)
        if len(seq_lens) > 2:
            log_seq_lens = np.log(seq_lens)
            log_times = np.log(seq_len_times)
            log_memory = np.log(seq_len_memory)
            
            time_scaling_coeff = np.polyfit(log_seq_lens, log_times, 1)[0]
            memory_scaling_coeff = np.polyfit(log_seq_lens, log_memory, 1)[0]
        else:
            time_scaling_coeff = None
            memory_scaling_coeff = None
        
        return {
            'sequence_length_scaling': {
                'seq_lens': seq_lens,
                'times': seq_len_times,
                'memory_usage': seq_len_memory,
                'time_scaling_exponent': time_scaling_coeff,
                'memory_scaling_exponent': memory_scaling_coeff,
            }
        }
    
    def print_results(self, module_name: str):
        """Print formatted benchmark results."""
        if module_name not in self.results:
            print(f"No results found for module: {module_name}")
            return
            
        results = self.results[module_name]
        stats = results['summary_stats']
        
        print(f"\n{'='*50}")
        print(f"Benchmark Results: {module_name}")
        print(f"{'='*50}")
        
        print(f"\nSummary Statistics:")
        print(f"  Total configurations tested: {stats['total_configs']}")
        print(f"  Average time: {stats['time_stats']['mean']:.4f}s ± {stats['time_stats']['std']:.4f}s")
        print(f"  Time range: {stats['time_stats']['min']:.4f}s - {stats['time_stats']['max']:.4f}s")
        print(f"  Average throughput: {stats['throughput_stats']['mean_tokens_per_sec']:.1f} tokens/sec")
        print(f"  Peak throughput: {stats['throughput_stats']['max_tokens_per_sec']:.1f} tokens/sec")
        print(f"  Average memory: {stats['memory_stats']['mean_memory_mb']:.1f} MB")
        print(f"  Peak memory: {stats['memory_stats']['max_memory_mb']:.1f} MB")
        
        # Show a few sample configurations
        print(f"\nSample Configuration Results:")
        sample_configs = results['configurations'][:3]  # First 3 configs
        
        for i, config in enumerate(sample_configs):
            print(f"  Config {i+1}: B={config['batch_size']}, S={config['seq_len']}, H={config['num_heads']}")
            print(f"    Time: {config['mean_time']:.4f}s")
            print(f"    Throughput: {config['tokens_per_second']:.1f} tokens/sec")
            print(f"    Memory: {config['memory_usage']['memory_delta_mb']:.1f} MB")
    
    def save_results(self, filepath: str):
        """Save benchmark results to file."""
        import json
        
        # Convert numpy types to native Python types for JSON serialization
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
        
        serializable_results = convert_numpy(self.results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to: {filepath}")