"""Integration tests for attention modules."""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Add the claude directory to Python path for benchmarks
claude_path = Path(__file__).parent
sys.path.insert(0, str(claude_path))

import jax
import jax.numpy as jnp
from flax import linen as nn

from openpi.models.gemma import Attention as GemmaAttention, Config as GemmaConfig
from openpi.models.gemma_fast import Attention as GemmaFastAttention

from benchmarks.performance_benchmark import AttentionPerformanceBenchmark
from benchmarks.memory_benchmark import AttentionMemoryBenchmark
from benchmarks.correctness_test import AttentionCorrectnessTest
from benchmarks.benchmark_config import BENCHMARK_CONFIG


class GemmaAttentionWrapper(nn.Module):
    """Wrapper for Gemma attention to make it compatible with benchmarking."""
    
    num_heads: int
    num_kv_heads: int
    head_dim: int
    width: int
    
    def setup(self):
        # Create a single config for the attention module
        self.config = GemmaConfig(
            depth=1,  # Number of layers (not used by attention module)
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            width=self.width,
            mlp_dim=self.width * 4,  # Standard 4x expansion
        )
        
        self.attention = GemmaAttention(configs=[self.config])
    
    def __call__(self, x, positions, attn_mask, kv_cache):
        # Gemma attention expects a list of inputs (for multiple experts)
        outputs, new_kv_cache = self.attention([x], positions, attn_mask, kv_cache)
        # Return single output (first expert)
        return outputs[0], new_kv_cache


class GemmaFastAttentionWrapper(nn.Module):
    """Wrapper for Gemma fast attention to make it compatible with benchmarking."""
    
    num_heads: int
    num_kv_heads: int
    head_dim: int
    width: int
    
    def setup(self):
        self.attention = GemmaFastAttention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            features=self.width
        )
    
    def __call__(self, x, positions, attn_mask, kv_cache):
        # GemmaFast attention requires a decode parameter
        # For benchmarking, we'll assume decode=False (prefill mode)
        decode = False
        return self.attention(x, positions, attn_mask, kv_cache, decode)


def create_attention_modules(config):
    """Create attention module instances for testing."""
    
    # Gemma attention (wrapped)
    gemma_attention = GemmaAttentionWrapper(
        num_heads=config['num_heads'],
        num_kv_heads=config['num_kv_heads'],
        head_dim=config['head_dim'],
        width=config['width']
    )
    
    # Gemma fast attention (wrapped)
    gemma_fast_attention = GemmaFastAttentionWrapper(
        num_heads=config['num_heads'],
        num_kv_heads=config['num_kv_heads'],
        head_dim=config['head_dim'],
        width=config['width']
    )
    
    return {
        'gemma': gemma_attention,
        'gemma_fast': gemma_fast_attention
    }


def run_performance_benchmarks():
    """Run performance benchmarks on attention implementations."""
    print("="*60)
    print("RUNNING PERFORMANCE BENCHMARKS")
    print("="*60)
    
    benchmark = AttentionPerformanceBenchmark()
    
    # Test with a subset of configurations for faster execution
    test_configs = [
        {'batch_size': 1, 'seq_len': 512, 'num_heads': 32, 'num_kv_heads': 8, 
         'head_dim': 256, 'width': 8192},
        {'batch_size': 4, 'seq_len': 1024, 'num_heads': 32, 'num_kv_heads': 8,
         'head_dim': 256, 'width': 8192},
        {'batch_size': 1, 'seq_len': 2048, 'num_heads': 32, 'num_kv_heads': 8,
         'head_dim': 256, 'width': 8192},
    ]
    
    modules = create_attention_modules(test_configs[0])
    
    # Benchmark each module
    for module_name, module in modules.items():
        print(f"\nBenchmarking {module_name}...")
        try:
            results = benchmark.benchmark_attention_module(
                module, module_name, test_configs
            )
            benchmark.print_results(module_name)
        except Exception as e:
            print(f"Error benchmarking {module_name}: {e}")
    
    # Compare modules
    if len(benchmark.results) > 1:
        print(f"\n" + "="*40)
        print("PERFORMANCE COMPARISON")
        print("="*40)
        
        try:
            comparison = benchmark.compare_modules(list(benchmark.results.keys()))
            
            baseline_name = list(benchmark.results.keys())[0]
            print(f"Using {baseline_name} as baseline:")
            
            for module_name in list(benchmark.results.keys())[1:]:
                rel_perf = comparison['relative_performance'][module_name]
                print(f"\n{module_name} vs {baseline_name}:")
                print(f"  Time ratio: {rel_perf['time_ratio']:.2f}x")
                print(f"  Throughput ratio: {rel_perf['throughput_ratio']:.2f}x")
                print(f"  Memory ratio: {rel_perf['memory_ratio']:.2f}x")
        except Exception as e:
            print(f"Error in comparison: {e}")
    
    # Save results
    try:
        benchmark.save_results('/root/openpi/claude/performance_results.json')
    except Exception as e:
        print(f"Error saving results: {e}")


def run_memory_benchmarks():
    """Run memory benchmarks on attention implementations."""
    print("\n" + "="*60)
    print("RUNNING MEMORY BENCHMARKS")
    print("="*60)
    
    benchmark = AttentionMemoryBenchmark()
    
    # Test with smaller configurations for memory testing
    test_configs = [
        {'batch_size': 1, 'seq_len': 512, 'num_heads': 32, 'num_kv_heads': 8,
         'head_dim': 256, 'width': 8192},
        {'batch_size': 1, 'seq_len': 1024, 'num_heads': 32, 'num_kv_heads': 8,
         'head_dim': 256, 'width': 8192},
        {'batch_size': 1, 'seq_len': 2048, 'num_heads': 32, 'num_kv_heads': 8,
         'head_dim': 256, 'width': 8192},
    ]
    
    modules = create_attention_modules(test_configs[0])
    
    # Benchmark each module
    for module_name, module in modules.items():
        print(f"\nMemory benchmarking {module_name}...")
        try:
            results = benchmark.benchmark_memory_usage(
                module, module_name, test_configs
            )
            benchmark.print_memory_results(module_name)
        except Exception as e:
            print(f"Error in memory benchmark for {module_name}: {e}")
    
    # Compare memory usage
    if len(benchmark.results) > 1:
        print(f"\n" + "="*40)
        print("MEMORY USAGE COMPARISON")
        print("="*40)
        
        try:
            comparison = benchmark.compare_memory_usage(list(benchmark.results.keys()))
            
            baseline_name = list(benchmark.results.keys())[0]
            print(f"Using {baseline_name} as baseline:")
            
            for module_name in list(benchmark.results.keys())[1:]:
                efficiency = comparison['efficiency_comparison'][module_name]
                print(f"\n{module_name} vs {baseline_name}:")
                print(f"  Relative memory usage: {efficiency['relative_memory_usage']:.2f}x")
                print(f"  Memory savings: {efficiency['memory_savings_mb']:.1f} MB ({efficiency['memory_savings_percent']:.1f}%)")
        except Exception as e:
            print(f"Error in memory comparison: {e}")
    
    # Save results
    try:
        benchmark.save_memory_results('/root/openpi/claude/memory_results.json')
    except Exception as e:
        print(f"Error saving memory results: {e}")


def run_correctness_tests():
    """Run correctness tests on attention implementations."""
    print("\n" + "="*60)
    print("RUNNING CORRECTNESS TESTS")
    print("="*60)
    
    tester = AttentionCorrectnessTest()
    
    # Test with small configurations for correctness
    test_configs = [
        {'batch_size': 1, 'seq_len': 512, 'num_heads': 32, 'num_kv_heads': 8,
         'head_dim': 256, 'width': 8192},
        {'batch_size': 2, 'seq_len': 1024, 'num_heads': 32, 'num_kv_heads': 8,
         'head_dim': 256, 'width': 8192},
    ]
    
    modules = create_attention_modules(test_configs[0])
    module_names = list(modules.keys())
    
    # Test numerical stability for each module
    for module_name, module in modules.items():
        print(f"\nTesting stability of {module_name}...")
        try:
            results = tester.test_numerical_stability(
                module, module_name, test_configs
            )
            tester.print_correctness_results(f"{module_name}_stability")
        except Exception as e:
            print(f"Error in stability test for {module_name}: {e}")
    
    # Compare implementations (if we have multiple)
    if len(modules) > 1:
        print(f"\n" + "="*40)
        print("CORRECTNESS COMPARISON")
        print("="*40)
        
        try:
            # Compare first two modules
            module1_name, module1 = list(modules.items())[0]
            module2_name, module2 = list(modules.items())[1]
            
            results = tester.test_attention_correctness(
                module1, module2, module1_name, module2_name, test_configs
            )
            tester.print_correctness_results(f"{module1_name}_vs_{module2_name}")
        except Exception as e:
            print(f"Error in correctness comparison: {e}")
    
    # Save results
    try:
        tester.save_correctness_results('/root/openpi/claude/correctness_results.json')
    except Exception as e:
        print(f"Error saving correctness results: {e}")


def main():
    """Main test runner."""
    print("Flash Attention Benchmarking Suite")
    print("==================================")
    
    # Check JAX backend
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    
    try:
        # Run all benchmark suites
        run_performance_benchmarks()
        run_memory_benchmarks() 
        run_correctness_tests()
        
        print("\n" + "="*60)
        print("BENCHMARKING COMPLETE")
        print("="*60)
        print("Results saved to:")
        print("  - performance_results.json")
        print("  - memory_results.json")  
        print("  - correctness_results.json")
        
    except Exception as e:
        print(f"\nBenchmarking failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()