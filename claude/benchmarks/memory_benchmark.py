"""Memory benchmarking for attention implementations."""

import time
import threading
from typing import Dict, List, Tuple, Any, Optional, Callable
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from .benchmark_config import BENCHMARK_CONFIG
from .utils import create_random_inputs, get_memory_usage, format_bytes


class MemoryProfiler:
    """Memory profiler for tracking JAX memory usage over time."""
    
    def __init__(self, sample_interval: float = 0.1):
        self.sample_interval = sample_interval
        self.memory_samples = []
        self.timestamps = []
        self.profiling = False
        self.profile_thread = None
        
    def start_profiling(self):
        """Start memory profiling in a separate thread."""
        if self.profiling:
            return
            
        self.profiling = True
        self.memory_samples = []
        self.timestamps = []
        self.profile_thread = threading.Thread(target=self._profile_loop)
        self.profile_thread.daemon = True
        self.profile_thread.start()
    
    def stop_profiling(self) -> Dict[str, Any]:
        """Stop memory profiling and return results."""
        if not self.profiling:
            return {}
            
        self.profiling = False
        if self.profile_thread:
            self.profile_thread.join(timeout=1.0)
        
        if not self.memory_samples:
            return {}
        
        memory_array = np.array(self.memory_samples)
        timestamps = np.array(self.timestamps)
        
        return {
            'timestamps': timestamps.tolist(),
            'memory_usage_bytes': memory_array.tolist(),
            'memory_usage_mb': (memory_array / (1024 * 1024)).tolist(),
            'peak_memory_bytes': int(np.max(memory_array)),
            'peak_memory_mb': float(np.max(memory_array) / (1024 * 1024)),
            'average_memory_bytes': int(np.mean(memory_array)),
            'average_memory_mb': float(np.mean(memory_array) / (1024 * 1024)),
            'memory_delta_bytes': int(np.max(memory_array) - np.min(memory_array)),
            'memory_delta_mb': float((np.max(memory_array) - np.min(memory_array)) / (1024 * 1024)),
            'sample_count': len(memory_array),
            'duration_seconds': float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0,
        }
    
    def _profile_loop(self):
        """Main profiling loop running in separate thread."""
        start_time = time.perf_counter()
        
        while self.profiling:
            try:
                memory_info = get_memory_usage()
                current_time = time.perf_counter()
                
                self.memory_samples.append(memory_info['used_memory'])
                self.timestamps.append(current_time - start_time)
                
                time.sleep(self.sample_interval)
            except Exception:
                # Continue profiling even if we can't get memory info
                continue


class AttentionMemoryBenchmark:
    """Memory benchmarking suite for attention implementations."""
    
    def __init__(self, config=BENCHMARK_CONFIG):
        self.config = config
        self.results = {}
        
    def benchmark_memory_usage(
        self, 
        attention_module: nn.Module,
        module_name: str,
        test_configs: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Benchmark memory usage of attention module across configurations.
        
        Args:
            attention_module: The attention module to benchmark
            module_name: Name identifier for the module  
            test_configs: Optional list of test configurations
            
        Returns:
            Dictionary with memory benchmark results
        """
        if test_configs is None:
            test_configs = self._generate_test_configs()
            
        results = {
            'module_name': module_name,
            'configurations': [],
            'memory_scaling': {},
            'peak_memory_analysis': {}
        }
        
        print(f"Memory benchmarking {module_name}...")
        
        for i, config in enumerate(test_configs):
            print(f"  Config {i+1}/{len(test_configs)}: "
                  f"B={config['batch_size']}, S={config['seq_len']}, "
                  f"H={config['num_heads']}/{config['num_kv_heads']}")
            
            config_result = self._benchmark_single_config_memory(attention_module, config)
            config_result.update(config)
            results['configurations'].append(config_result)
            
        # Analyze memory scaling patterns
        results['memory_scaling'] = self._analyze_memory_scaling(results['configurations'])
        results['peak_memory_analysis'] = self._analyze_peak_memory(results['configurations'])
        
        self.results[module_name] = results
        return results
    
    def _generate_test_configs(self) -> List[Dict]:
        """Generate test configurations from benchmark config."""
        configs = []
        
        for seq_len in self.config.SEQUENCE_LENGTHS:
            for batch_size in self.config.BATCH_SIZES:
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
    
    def _benchmark_single_config_memory(self, attention_module: nn.Module, config: Dict) -> Dict[str, Any]:
        """Benchmark memory usage for a single configuration."""
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
            dummy_input = inputs['x'][:1, :1]
            variables = attention_module.init(
                key,
                dummy_input,
                inputs['positions'][:1, :1], 
                inputs['attn_mask'][:1, :, :1, :1],
                None
            )
            attention_module = attention_module.bind(variables)
        
        # Memory profiling during forward pass
        profiler = MemoryProfiler(sample_interval=self.config.MEMORY_SAMPLE_INTERVAL)
        
        # Get baseline memory
        baseline_memory = get_memory_usage()
        
        # Start profiling
        profiler.start_profiling()
        
        try:
            # Run forward pass
            result = attention_module(
                inputs['x'],
                inputs['positions'],
                inputs['attn_mask'], 
                None
            )
            
            # Ensure computation is complete
            if hasattr(result, 'block_until_ready'):
                result.block_until_ready()
            elif isinstance(result, (tuple, list)):
                for r in result:
                    if hasattr(r, 'block_until_ready'):
                        r.block_until_ready()
        finally:
            # Stop profiling
            profile_results = profiler.stop_profiling()
        
        # Get final memory usage
        final_memory = get_memory_usage()
        
        # Calculate theoretical memory requirements  
        theoretical_memory = self._calculate_theoretical_memory(config)
        
        # Combine results
        memory_results = {
            'baseline_memory': baseline_memory,
            'final_memory': final_memory,
            'memory_delta_bytes': final_memory['used_memory'] - baseline_memory['used_memory'],
            'memory_delta_mb': (final_memory['used_memory'] - baseline_memory['used_memory']) / (1024 * 1024),
            'theoretical_memory': theoretical_memory,
            'memory_efficiency': self._calculate_memory_efficiency(profile_results, theoretical_memory),
            'profile_data': profile_results,
        }
        
        return memory_results
    
    def _calculate_theoretical_memory(self, config: Dict) -> Dict[str, Any]:
        """Calculate theoretical memory requirements for attention."""
        batch_size = config['batch_size']
        seq_len = config['seq_len']
        num_heads = config['num_heads']
        head_dim = config['head_dim']
        width = config['width']
        
        # Bytes per element (float16 = 2 bytes, float32 = 4 bytes)
        bytes_per_element = 2 if self.config.DTYPE == 'float16' else 4
        
        # Input tensor: B x S x W
        input_memory = batch_size * seq_len * width * bytes_per_element
        
        # QKV tensors: B x S x H x D each
        qkv_memory = 3 * batch_size * seq_len * num_heads * head_dim * bytes_per_element
        
        # Attention matrix: B x H x S x S (this is the big one!)
        attention_matrix_memory = batch_size * num_heads * seq_len * seq_len * 4  # Usually float32
        
        # Output tensor: B x S x W
        output_memory = batch_size * seq_len * width * bytes_per_element
        
        # KV cache (if applicable): 2 x B x S x H x D
        kv_cache_memory = 2 * batch_size * seq_len * num_heads * head_dim * bytes_per_element
        
        total_memory = (input_memory + qkv_memory + attention_matrix_memory + 
                       output_memory + kv_cache_memory)
        
        return {
            'input_memory_bytes': input_memory,
            'qkv_memory_bytes': qkv_memory,
            'attention_matrix_memory_bytes': attention_matrix_memory,
            'output_memory_bytes': output_memory, 
            'kv_cache_memory_bytes': kv_cache_memory,
            'total_theoretical_bytes': total_memory,
            'total_theoretical_mb': total_memory / (1024 * 1024),
            'attention_matrix_dominance': attention_matrix_memory / total_memory,
        }
    
    def _calculate_memory_efficiency(self, profile_data: Dict, theoretical: Dict) -> Dict[str, Any]:
        """Calculate memory efficiency metrics.""" 
        if not profile_data or 'peak_memory_bytes' not in profile_data:
            return {'efficiency_ratio': 0.0, 'overhead_bytes': 0, 'overhead_ratio': 0.0}
        
        actual_peak = profile_data['peak_memory_bytes']
        theoretical_total = theoretical['total_theoretical_bytes']
        
        if theoretical_total == 0:
            return {'efficiency_ratio': 0.0, 'overhead_bytes': 0, 'overhead_ratio': 0.0}
        
        efficiency_ratio = theoretical_total / actual_peak if actual_peak > 0 else 0.0
        overhead_bytes = max(0, actual_peak - theoretical_total)
        overhead_ratio = overhead_bytes / theoretical_total if theoretical_total > 0 else 0.0
        
        return {
            'efficiency_ratio': efficiency_ratio,
            'overhead_bytes': overhead_bytes,
            'overhead_mb': overhead_bytes / (1024 * 1024),
            'overhead_ratio': overhead_ratio,
        }
    
    def _analyze_memory_scaling(self, configurations: List[Dict]) -> Dict[str, Any]:
        """Analyze how memory usage scales with different parameters."""
        # Group by sequence length for scaling analysis
        seq_len_groups = {}
        for config in configurations:
            seq_len = config['seq_len']
            if seq_len not in seq_len_groups:
                seq_len_groups[seq_len] = []
            seq_len_groups[seq_len].append(config)
        
        # Analyze sequence length scaling
        seq_lens = sorted(seq_len_groups.keys())
        seq_len_memory = []
        theoretical_memory = []
        
        for seq_len in seq_lens:
            configs = seq_len_groups[seq_len]
            avg_memory = np.mean([c['memory_delta_mb'] for c in configs])
            avg_theoretical = np.mean([c['theoretical_memory']['total_theoretical_mb'] for c in configs])
            seq_len_memory.append(avg_memory)
            theoretical_memory.append(avg_theoretical)
        
        # Fit scaling curve (attention should be O(n²))
        if len(seq_lens) > 2:
            # Linear fit in log space
            log_seq_lens = np.log(seq_lens)
            log_memory = np.log(seq_len_memory)
            log_theoretical = np.log(theoretical_memory)
            
            actual_scaling_coeff = np.polyfit(log_seq_lens, log_memory, 1)[0]
            theoretical_scaling_coeff = np.polyfit(log_seq_lens, log_theoretical, 1)[0]
        else:
            actual_scaling_coeff = None
            theoretical_scaling_coeff = None
        
        return {
            'sequence_length_scaling': {
                'seq_lens': seq_lens,
                'actual_memory_mb': seq_len_memory,
                'theoretical_memory_mb': theoretical_memory,
                'actual_scaling_exponent': actual_scaling_coeff,
                'theoretical_scaling_exponent': theoretical_scaling_coeff,
                'expected_quadratic': abs(theoretical_scaling_coeff - 2.0) < 0.2 if theoretical_scaling_coeff else None,
            }
        }
    
    def _analyze_peak_memory(self, configurations: List[Dict]) -> Dict[str, Any]:
        """Analyze peak memory usage patterns."""
        peak_memories = []
        theoretical_memories = []
        attention_matrix_sizes = []
        
        for config in configurations:
            peak_mb = config.get('memory_delta_mb', 0)
            theoretical_mb = config['theoretical_memory']['total_theoretical_mb']
            attention_matrix_mb = config['theoretical_memory']['attention_matrix_memory_bytes'] / (1024 * 1024)
            
            peak_memories.append(peak_mb)
            theoretical_memories.append(theoretical_mb)
            attention_matrix_sizes.append(attention_matrix_mb)
        
        peak_memories = np.array(peak_memories)
        theoretical_memories = np.array(theoretical_memories)
        attention_matrix_sizes = np.array(attention_matrix_sizes)
        
        return {
            'peak_memory_stats': {
                'max_peak_mb': float(np.max(peak_memories)),
                'min_peak_mb': float(np.min(peak_memories)),
                'mean_peak_mb': float(np.mean(peak_memories)),
                'std_peak_mb': float(np.std(peak_memories)),
            },
            'memory_breakdown': {
                'attention_matrix_dominance': float(np.mean(attention_matrix_sizes / theoretical_memories)),
                'largest_attention_matrix_mb': float(np.max(attention_matrix_sizes)),
                'attention_memory_ratio': float(np.mean(attention_matrix_sizes / peak_memories)),
            },
            'bottleneck_analysis': {
                'memory_bound_threshold_mb': 1000,  # Configurable threshold
                'memory_bound_configs': int(np.sum(peak_memories > 1000)),
                'attention_dominant_configs': int(np.sum(attention_matrix_sizes / theoretical_memories > 0.5)),
            }
        }
    
    def print_memory_results(self, module_name: str):
        """Print formatted memory benchmark results."""
        if module_name not in self.results:
            print(f"No memory results found for module: {module_name}")
            return
        
        results = self.results[module_name]
        scaling = results['memory_scaling']['sequence_length_scaling']
        peak_analysis = results['peak_memory_analysis']
        
        print(f"\n{'='*50}")
        print(f"Memory Benchmark Results: {module_name}")
        print(f"{'='*50}")
        
        print(f"\nMemory Scaling Analysis:")
        if scaling['actual_scaling_exponent']:
            print(f"  Actual scaling exponent: {scaling['actual_scaling_exponent']:.2f}")
            print(f"  Theoretical scaling exponent: {scaling['theoretical_scaling_exponent']:.2f}")
            print(f"  Expected quadratic scaling: {scaling['expected_quadratic']}")
        
        print(f"\nPeak Memory Statistics:")
        stats = peak_analysis['peak_memory_stats']
        print(f"  Maximum peak memory: {format_bytes(stats['max_peak_mb'] * 1024 * 1024)}")
        print(f"  Average peak memory: {format_bytes(stats['mean_peak_mb'] * 1024 * 1024)}")
        print(f"  Memory usage std dev: {format_bytes(stats['std_peak_mb'] * 1024 * 1024)}")
        
        print(f"\nMemory Breakdown:")
        breakdown = peak_analysis['memory_breakdown']
        print(f"  Attention matrix dominance: {breakdown['attention_matrix_dominance']:.1%}")
        print(f"  Largest attention matrix: {format_bytes(breakdown['largest_attention_matrix_mb'] * 1024 * 1024)}")
        print(f"  Attention/total memory ratio: {breakdown['attention_memory_ratio']:.1%}")
        
        print(f"\nBottleneck Analysis:")
        bottleneck = peak_analysis['bottleneck_analysis']
        print(f"  Memory-bound configs (>1GB): {bottleneck['memory_bound_configs']}")
        print(f"  Attention-dominant configs: {bottleneck['attention_dominant_configs']}")
        
        # Show scaling data
        print(f"\nSequence Length Scaling:")
        for i, seq_len in enumerate(scaling['seq_lens'][:5]):  # Show first 5
            actual = scaling['actual_memory_mb'][i]
            theoretical = scaling['theoretical_memory_mb'][i]
            print(f"  Seq len {seq_len}: {actual:.1f} MB actual, {theoretical:.1f} MB theoretical")
    
    def compare_memory_usage(self, module_names: List[str]) -> Dict[str, Any]:
        """Compare memory usage between multiple benchmarked modules."""
        if not all(name in self.results for name in module_names):
            missing = [name for name in module_names if name not in self.results]
            raise ValueError(f"Missing memory results for modules: {missing}")
        
        comparison = {
            'modules': module_names,
            'peak_memory_comparison': {},
            'scaling_comparison': {},
            'efficiency_comparison': {}
        }
        
        # Compare peak memory usage
        for module_name in module_names:
            results = self.results[module_name]
            peak_stats = results['peak_memory_analysis']['peak_memory_stats']
            scaling = results['memory_scaling']['sequence_length_scaling']
            
            comparison['peak_memory_comparison'][module_name] = {
                'max_peak_mb': peak_stats['max_peak_mb'],
                'mean_peak_mb': peak_stats['mean_peak_mb'],
            }
            
            comparison['scaling_comparison'][module_name] = {
                'scaling_exponent': scaling.get('actual_scaling_exponent'),
                'quadratic_behavior': scaling.get('expected_quadratic'),
            }
        
        # Calculate relative efficiency
        baseline_name = module_names[0]
        baseline_peak = comparison['peak_memory_comparison'][baseline_name]['mean_peak_mb']
        
        for module_name in module_names:
            module_peak = comparison['peak_memory_comparison'][module_name]['mean_peak_mb']
            comparison['efficiency_comparison'][module_name] = {
                'relative_memory_usage': module_peak / baseline_peak if baseline_peak > 0 else 1.0,
                'memory_savings_mb': baseline_peak - module_peak,
                'memory_savings_percent': ((baseline_peak - module_peak) / baseline_peak * 100) if baseline_peak > 0 else 0.0,
            }
        
        return comparison
    
    def save_memory_results(self, filepath: str):
        """Save memory benchmark results to file."""
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
        
        serializable_results = convert_numpy(self.results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Memory results saved to: {filepath}")