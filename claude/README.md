# Flash Attention Implementation - Phase 1: Benchmarking Infrastructure

This directory contains the benchmarking infrastructure for the Flash Attention implementation project. The goal is to establish comprehensive performance baselines before implementing flash attention optimizations.

## Overview

The benchmarking suite consists of three main components:

1. **Performance Benchmarking** - Wall-clock time, throughput, and compute efficiency
2. **Memory Benchmarking** - Memory usage patterns, scaling behavior, and bottleneck analysis  
3. **Correctness Testing** - Numerical precision validation and stability testing

## File Structure

```
claude/
├── benchmarks/
│   ├── __init__.py
│   ├── benchmark_config.py      # Configuration constants
│   ├── performance_benchmark.py # Performance benchmarking suite
│   ├── memory_benchmark.py      # Memory usage analysis
│   ├── correctness_test.py      # Correctness validation
│   └── utils.py                 # Shared utilities
├── test_attention.py            # Integration tests
├── run_benchmarks.py            # Main benchmark runner
├── PHASE1_IMPLEMENTATION_PLAN.md
└── README.md                    # This file
```

## Quick Start

### Running Benchmarks

```bash
# Activate the virtual environment first
cd /root/openpi
source .venv/bin/activate

# Run all benchmarks with default settings
cd claude
python run_benchmarks.py

# Run quick benchmarks (reduced configurations for faster execution)
python run_benchmarks.py --quick

# Test specific modules only
python run_benchmarks.py --modules gemma gemma_fast

# Specify output directory
python run_benchmarks.py --output-dir ./my_results

# Or run directly with .venv python
cd /root/openpi
.venv/bin/python claude/run_benchmarks.py --quick
```

### Running Individual Tests

```bash
# Make sure to use the virtual environment
cd /root/openpi
source .venv/bin/activate

# Run integration tests
cd claude
python test_attention.py

# Or run directly with .venv python
cd /root/openpi
.venv/bin/python claude/test_attention.py

# Import and use benchmarking components directly
.venv/bin/python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('src')))

from claude.benchmarks.performance_benchmark import AttentionPerformanceBenchmark
from claude.test_attention import create_attention_modules

config = {'num_heads': 32, 'num_kv_heads': 8, 'head_dim': 256, 'width': 8192}
modules = create_attention_modules(config)
benchmark = AttentionPerformanceBenchmark()

for name, module in modules.items():
    results = benchmark.benchmark_attention_module(module, name)
    benchmark.print_results(name)
"
```

## Benchmark Configuration

Key configuration parameters in `benchmarks/benchmark_config.py`:

```python
# Test configurations from implementation plan
SEQUENCE_LENGTHS = (512, 1024, 2048, 4096, 8192)
BATCH_SIZES = (1, 4, 8, 16) 
HEAD_CONFIGS = ((32, 32), (32, 8))  # (num_heads, num_kv_heads)

# Model parameters
HEAD_DIM = 256
WIDTH = 8192  # Model width/embedding dimension

# Benchmark settings
WARMUP_RUNS = 3  # XLA compilation warmup
BENCHMARK_RUNS = 10  # Statistical averaging
```

## Understanding Results

### Performance Results

- **Wall-clock time**: Actual execution time including all overheads
- **Throughput**: Tokens processed per second
- **FLOPS**: Floating point operations per second (theoretical)
- **Memory delta**: Additional memory used during computation

### Memory Results  

- **Peak memory usage**: Maximum memory consumed during attention
- **Theoretical vs Actual**: Comparison with calculated memory requirements
- **Scaling exponent**: How memory scales with sequence length (should be ~2.0 for quadratic)
- **Attention matrix dominance**: Fraction of memory used by attention matrix

### Correctness Results

- **Determinism**: Same inputs produce same outputs
- **Numerical stability**: No NaN/Inf values across input scales
- **Gradient computation**: Gradients can be computed without errors
- **Output comparison**: Numerical differences between implementations

## Expected Baseline Performance

Based on the implementation plan, current attention implementations should show:

- **Memory scaling**: O(n²) with sequence length due to attention matrix
- **Compute scaling**: O(n²) operations for attention computation  
- **Memory bottleneck**: Attention matrix (B × H × S × S) dominates memory usage
- **Performance**: Reasonable throughput for sequences up to 2K tokens

## Using Results for Flash Attention

The benchmark results establish baselines for:

1. **Performance targets**: Flash attention should improve wall-clock time by 15-25% for long sequences
2. **Memory targets**: 50-80% reduction in attention memory usage  
3. **Correctness validation**: Flash attention outputs must match baseline within tolerance
4. **Scaling analysis**: Identify sequence length thresholds where flash attention provides benefits

## Integration with Existing Code

The benchmarking infrastructure integrates with existing attention modules:

- `src/openpi/models/gemma.py:150-242` - Multi-expert Gemma attention
- `src/openpi/models/gemma_fast.py:125-225` - Fast Gemma attention with KV caching

Modules are wrapped to provide consistent interfaces for benchmarking.

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure the `src` directory is in Python path
2. **Memory errors**: Reduce batch sizes or sequence lengths in config
3. **JAX compilation time**: First runs may be slow due to XLA compilation
4. **Device issues**: Check `jax.devices()` shows expected GPU/TPU

### Performance Tips

- Use `--quick` flag for faster testing during development
- Monitor GPU memory usage with `nvidia-smi` during benchmarks
- JAX compilation caching helps with repeated runs
- Consider smaller configurations for rapid iteration

## Next Steps

After establishing baselines with this infrastructure:

1. **Analyze bottlenecks**: Identify where attention is memory/compute bound
2. **Select flash attention library**: Evaluate Kvax, CuDNN attention, etc.
3. **Implement integration**: Add flash attention as configurable option
4. **Validate improvements**: Use same benchmarking suite to measure gains
5. **Optimize thresholds**: Find optimal sequence lengths for flash attention activation

This benchmarking infrastructure will be reused throughout the flash attention implementation to ensure improvements are real and regressions are caught early.