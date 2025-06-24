#!/usr/bin/env python3
"""
Script to inspect captured attention inputs.
"""

import numpy as np
import os
import glob

def inspect_captured_inputs():
    """Inspect all captured attention input files."""
    capture_dir = "/root/openpi/claude/attention_captures"
    
    if not os.path.exists(capture_dir):
        print("No attention captures directory found.")
        return
    
    files = glob.glob(os.path.join(capture_dir, "*.npz"))
    if not files:
        print("No captured files found.")
        return
    
    print(f"Found {len(files)} captured attention input files:")
    
    for file_path in sorted(files):
        print(f"\n=== {os.path.basename(file_path)} ===")
        
        # Load the data
        data = np.load(file_path, allow_pickle=True)
        
        # Print basic info
        print("Keys:", list(data.keys()))
        
        # Print metadata
        if 'metadata' in data:
            metadata = data['metadata'].item()  # Extract dict from numpy array
            print("Metadata:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
        
        # Print tensor shapes
        for key in data.keys():
            if key != 'metadata' and data[key] is not None:
                if hasattr(data[key], 'shape'):
                    print(f"{key} shape: {data[key].shape}")
                else:
                    print(f"{key}: {data[key]}")
        
        data.close()

if __name__ == "__main__":
    inspect_captured_inputs()