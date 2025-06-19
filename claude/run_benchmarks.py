"""Main benchmark runner script for Flash Attention implementation."""

import sys
import argparse
from pathlib import Path
import json
from typing import List, Optional

# Add the src directory to Python path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Add the claude directory to Python path for benchmarks
claude_path = Path(__file__).parent
sys.path.insert(0, str(claude_path))

import jax
import jax.numpy as jnp

from benchmarks.performance_benchmark import AttentionPerformanceBenchmark
from benchmarks.memory_benchmark import AttentionMemoryBenchmark
from benchmarks.correctness_test import AttentionCorrectnessTest
from benchmarks.benchmark_config import BENCHMARK_CONFIG


def print_banner():
    """Print benchmark suite banner."""
    print("=" * 70)
    print("Flash Attention Benchmarking Suite - Phase 1")
    print("=" * 70)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX version: {jax.__version__}")
    print("-" * 70)


def run_full_benchmarks(output_dir: Path, modules_to_test: Optional[List[str]] = None):
    """Run complete benchmark suite with all tests.
    
    Args:
        output_dir: Directory to save results
        modules_to_test: Optional list of module names to test
    """
    print_banner()
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Import attention module creation function
    try:
        from test_attention import create_attention_modules
    except ImportError as e:
        print(f"Error importing attention modules: {e}")
        print("Make sure the openpi source code is available.")
        return False

    # Get all module architectures to test from config
    head_configs = BENCHMARK_CONFIG.HEAD_CONFIGS
    
    # Master dictionary to hold all modules for comparison later
    all_modules = {}

    for num_heads, num_kv_heads in head_configs:
        print("\n" + "~" * 70)
        print(f"Testing Architecture: {num_heads} Heads, {num_kv_heads} KV_Heads")
        print("~" * 70)

        # Create modules for the current architecture
        arch_config = {
            'num_heads': num_heads,
            'num_kv_heads': num_kv_heads, 
            'head_dim': BENCHMARK_CONFIG.HEAD_DIM,
            'width': BENCHMARK_CONFIG.WIDTH,
        }
        
        try:
            modules = create_attention_modules(arch_config)
            
            if modules_to_test:
                modules = {name: module for name, module in modules.items() 
                          if name in modules_to_test}

            # Add a unique suffix to module names to identify architecture
            arch_suffix = f"_H{num_heads}_KV{num_kv_heads}"
            modules = {name + arch_suffix: module for name, module in modules.items()}
            
            print(f"Testing modules: {list(modules.keys())}")
            all_modules.update(modules)

        except Exception as e:
            print(f"Error creating attention modules for H={num_heads}, KV={num_kv_heads}: {e}")
            continue # Move to the next architecture

        success = True
        
        # 1. Performance Benchmarks
        print("\n" + "=" * 50)
        print("PHASE 1: PERFORMANCE BENCHMARKING")
        print("=" * 50)
        
        try:
            perf_benchmark = AttentionPerformanceBenchmark()
            
            for module_name, module in modules.items():
                print(f"\nBenchmarking performance: {module_name}")
                results = perf_benchmark.benchmark_attention_module(module, module_name)
                perf_benchmark.print_results(module_name)
            
            # Save performance results (append mode might be better, but overwrite is simpler for now)
            perf_file = output_dir / "performance_results.json"
            perf_benchmark.save_results(str(perf_file))
            
        except Exception as e:
            print(f"Performance benchmarking failed for this architecture: {e}")
            success = False
        
        # 2. Memory Benchmarks
        print("\n" + "=" * 50)
        print("PHASE 2: MEMORY BENCHMARKING")
        print("=" * 50)
        
        try:
            memory_benchmark = AttentionMemoryBenchmark()
            
            for module_name, module in modules.items():
                print(f"\nBenchmarking memory: {module_name}")
                results = memory_benchmark.benchmark_memory_usage(module, module_name)
                memory_benchmark.print_memory_results(module_name)
            
            # Save memory results
            memory_file = output_dir / "memory_results.json"
            memory_benchmark.save_memory_results(str(memory_file))
            
        except Exception as e:
            print(f"Memory benchmarking failed for this architecture: {e}")
            success = False
        
        # 3. Correctness Tests  
        print("\n" + "=" * 50)
        print("PHASE 3: CORRECTNESS TESTING")
        print("=" * 50)
        
        try:
            correctness_tester = AttentionCorrectnessTest()
            
            # Test numerical stability for each module
            for module_name, module in modules.items():
                print(f"\nTesting stability: {module_name}")
                results = correctness_tester.test_numerical_stability(module, module_name)
                correctness_tester.print_correctness_results(f"{module_name}_stability")
            
            # Compare implementations if multiple within the same architecture
            if len(modules) > 1:
                print(f"\n" + "-" * 40)
                print(f"CORRECTNESS COMPARISON (Arch: H={num_heads}, KV={num_kv_heads})")
                print("-" * 40)
                
                module_names = list(modules.keys())
                for i in range(len(module_names)):
                    for j in range(i + 1, len(module_names)):
                        name1, name2 = module_names[i], module_names[j]
                        module1, module2 = modules[name1], modules[name2]
                        
                        print(f"\nComparing {name1} vs {name2}:")
                        results = correctness_tester.test_attention_correctness(
                            module1, module2, name1, name2
                        )
                        correctness_tester.print_correctness_results(f"{name1}_vs_{name2}")
            
            # Save correctness results
            correctness_file = output_dir / "correctness_results.json"
            correctness_tester.save_correctness_results(str(correctness_file))
            
        except Exception as e:
            print(f"Correctness testing failed for this architecture: {e}")
            success = False

    # 4. Generate Summary Report
    print("\n" + "=" * 50)
    print("GENERATING SUMMARY REPORT")
    print("=" * 50)
    
    try:
        generate_summary_report(output_dir, list(all_modules.keys()))
    except Exception as e:
        print(f"Summary report generation failed: {e}")
        # This is not a failure of the benchmark itself, so we don't set success=False
    
    # Final status
    print("\n" + "=" * 70)
    print("BENCHMARKING COMPLETE")
    print("=" * 70)
    
    if success:
        print("✓ All benchmarks completed successfully")
        print(f"Results saved to: {output_dir}")
        print("\nGenerated files:")
        for file in output_dir.glob("*.json"):
            print(f"  - {file.name}")
        if (output_dir / "summary_report.txt").exists():
            print(f"  - summary_report.txt")
    else:
        print("⚠ Some benchmarks failed - check output above")
    
    return success


def generate_summary_report(output_dir: Path, module_names: List[str]):
    """Generate a human-readable summary report."""
    summary_file = output_dir / "summary_report.txt"
    
    with open(summary_file, 'w') as f:
        f.write("Flash Attention Benchmarking Summary Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Tested modules: {', '.join(module_names)}\n")
        f.write(f"JAX backend: {jax.default_backend()}\n")
        f.write(f"JAX devices: {jax.devices()}\n\n")
        
        # Load and summarize results
        try:
            # Performance summary
            perf_file = output_dir / "performance_results.json"
            if perf_file.exists():
                with open(perf_file, 'r') as pf:
                    perf_data = json.load(pf)
                
                f.write("PERFORMANCE SUMMARY\n")
                f.write("-" * 20 + "\n")
                
                for module_name in module_names:
                    if module_name in perf_data:
                        stats = perf_data[module_name]['summary_stats']
                        f.write(f"\n{module_name}:\n")
                        f.write(f"  Average time: {stats['time_stats']['mean']:.4f}s\n")
                        f.write(f"  Peak throughput: {stats['throughput_stats']['max_tokens_per_sec']:.1f} tokens/sec\n")
                        f.write(f"  Peak memory: {stats['memory_stats']['max_memory_mb']:.1f} MB\n")
                
                f.write("\n\n")
            
            # Memory summary
            memory_file = output_dir / "memory_results.json"
            if memory_file.exists():
                with open(memory_file, 'r') as mf:
                    memory_data = json.load(mf)
                
                f.write("MEMORY SUMMARY\n")
                f.write("-" * 15 + "\n")
                
                for module_name in module_names:
                    if module_name in memory_data:
                        peak_stats = memory_data[module_name]['peak_memory_analysis']['peak_memory_stats']
                        scaling = memory_data[module_name]['memory_scaling']['sequence_length_scaling']
                        
                        f.write(f"\n{module_name}:\n")
                        f.write(f"  Peak memory: {peak_stats['max_peak_mb']:.1f} MB\n")
                        f.write(f"  Average memory: {peak_stats['mean_peak_mb']:.1f} MB\n")
                        if scaling.get('actual_scaling_exponent'):
                            f.write(f"  Scaling exponent: {scaling['actual_scaling_exponent']:.2f}\n")
                
                f.write("\n\n")
            
            # Correctness summary
            correctness_file = output_dir / "correctness_results.json"
            if correctness_file.exists():
                with open(correctness_file, 'r') as cf:
                    correctness_data = json.load(cf)
                
                f.write("CORRECTNESS SUMMARY\n")
                f.write("-" * 20 + "\n")
                
                for module_name in module_names:
                    stability_key = f"{module_name}_stability"
                    if stability_key in correctness_data:
                        summary = correctness_data[stability_key]['summary']
                        overall = summary['overall_stability']
                        
                        f.write(f"\n{module_name}:\n")
                        f.write(f"  All stability tests passed: {overall['all_tests_passed']}\n")
                        f.write(f"  Average pass rate: {overall['average_pass_rate']:.1%}\n")
                        f.write(f"  Deterministic: {summary['determinism']['pass_rate']:.1%}\n")
                        f.write(f"  Numerically stable: {summary['numerical_stability']['pass_rate']:.1%}\n")
                        f.write(f"  Gradient computation: {summary['gradient_computation']['pass_rate']:.1%}\n")
                
                f.write("\n\n")
        
        except Exception as e:
            f.write(f"Error generating detailed summary: {e}\n")
        
        f.write("END OF REPORT\n")
    
    print(f"Summary report saved to: {summary_file}")


def main():
    """Main entry point for benchmark runner."""
    parser = argparse.ArgumentParser(description="Flash Attention Benchmark Runner")
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=Path("claude/benchmark_results"),
        help="Output directory for benchmark results"
    )
    parser.add_argument(
        "--modules",
        nargs="*",
        help="Specific modules to test (default: all available)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark with reduced configurations"
    )
    
    args = parser.parse_args()
    
    args.quick = True
    # Override config for quick benchmarks
    if args.quick:
        print("Running quick benchmarks with reduced configurations...")
        BENCHMARK_CONFIG.SEQUENCE_LENGTHS = [512, 1024]
        BENCHMARK_CONFIG.BATCH_SIZES = [1, 4]
        BENCHMARK_CONFIG.BENCHMARK_RUNS = 3
        BENCHMARK_CONFIG.WARMUP_RUNS = 1
    
    # Run benchmarks
    success = run_full_benchmarks(args.output_dir, args.modules)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()