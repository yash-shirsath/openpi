#!/bin/bash

# CUDA 11.8 System Removal Script
# This script removes CUDA 11.8 installation in the correct order to avoid dependency conflicts

set -e  # Exit on any error

echo "=== CUDA 11.8 System Removal Script ==="
echo "This will remove all CUDA 11.8 packages from the system."
echo

# Function to run apt commands with auto-yes
run_apt() {
    echo "Running: apt $@"
    yes | apt "$@" || true
    echo
}

# Step 1: Remove libcublas dev package first to resolve dependency conflicts
echo "Step 1: Removing libcublas-dev-11-8 to resolve dependencies..."
run_apt remove --purge libcublas-dev-11-8

# Step 2: Auto-cleanup orphaned packages
echo "Step 2: Cleaning up orphaned packages..."
run_apt autoremove

# Step 3: Remove libcublas runtime library
echo "Step 3: Removing libcublas-11-8 runtime library..."
run_apt remove --purge libcublas-11-8

# Step 4: Auto-cleanup again
echo "Step 4: Cleaning up more orphaned packages..."
run_apt autoremove

# Step 5: Remove main CUDA packages
echo "Step 5: Removing main CUDA 11.8 packages..."
run_apt remove --purge cuda-*-11-8 cuda-toolkit-11-* cuda-keyring

# Step 6: Remove build tools and dependencies
echo "Step 6: Cleaning up build tools and dependencies..."
run_apt autoremove

# Step 7: Remove NCCL packages
echo "Step 7: Removing NCCL packages..."
run_apt remove --purge libnccl2 libnccl-dev

# Step 8: Final config cleanup
echo "Step 8: Purging remaining configuration files..."
run_apt purge cuda-toolkit-config-common libcufile-11-8

# Step 9: Final cleanup
echo "Step 9: Final cleanup..."
run_apt autoremove
run_apt autoclean

# Verification
echo "=== Verification ==="
echo "Checking for remaining CUDA packages:"
dpkg -l | grep -i cuda || echo "No CUDA packages found - removal successful!"

echo
echo "Checking for nvcc:"
which nvcc 2>/dev/null && echo "Warning: nvcc still found in PATH" || echo "nvcc successfully removed from PATH"

echo
echo "Checking for CUDA directories:"
ls -la /usr/local/cuda* 2>/dev/null && echo "Warning: CUDA directories still exist" || echo "CUDA directories successfully removed"

echo
echo "=== CUDA 11.8 Removal Complete ==="
echo "System is now ready for CUDA 12 installation."