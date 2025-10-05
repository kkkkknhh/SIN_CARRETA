"""
Quick test for CUDA device configuration.
"""

import torch
from device_config import initialize_device_config


def test_cuda_config():
    """Test CUDA configuration and thread settings."""
    print("=== CUDA Configuration Test ===")
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    
    # Test CUDA configuration
    config = initialize_device_config('cuda')
    device = config.get_device()
    threads = torch.get_num_threads()
    
    print(f"Configured device: {device}")
    print(f"PyTorch threads: {threads}")
    
    if device.type == 'cuda':
        print("✓ Successfully configured CUDA device")
        print(f"✓ PyTorch threads set to {threads} (should be 1 for CUDA)")
    else:
        print("✓ Fallback to CPU (CUDA not available)")
        print(f"✓ PyTorch threads: {threads}")
    
    # Test CPU configuration
    config_cpu = initialize_device_config('cpu')
    device_cpu = config_cpu.get_device()
    threads_cpu = torch.get_num_threads()
    
    print(f"\nCPU device: {device_cpu}")
    print(f"CPU threads: {threads_cpu}")


if __name__ == "__main__":
    test_cuda_config()