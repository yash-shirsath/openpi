"""Configuration constants for attention benchmarking."""

from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class BenchmarkConfig:
    """Configuration for attention benchmarking."""
    
    # Test configurations from implementation plan
    SEQUENCE_LENGTHS: List[int] = (512, 1024, 2048, 4096, 8192)
    BATCH_SIZES: List[int] = (1, 4, 8, 16) 
    HEAD_CONFIGS: List[Tuple[int, int]] = ((32, 32), (32, 8))  # (num_heads, num_kv_heads)
    
    # Model parameters
    HEAD_DIM: int = 256
    WIDTH: int = 8192  # Model width/embedding dimension
    
    # Benchmark settings
    WARMUP_RUNS: int = 3  # XLA compilation warmup
    BENCHMARK_RUNS: int = 10  # Statistical averaging
    
    # Memory tracking
    MEMORY_SAMPLE_INTERVAL: float = 0.1  # seconds
    
    # Device settings
    DTYPE: str = "float16"  # Match typical model precision
    
    # Attention-specific
    USE_CAUSAL_MASK: bool = True
    MAX_CACHE_SIZE: int = 8192


# Global benchmark configuration instance
BENCHMARK_CONFIG = BenchmarkConfig()