"""
Test script for device configuration functionality.
"""

import torch

from device_config import DeviceConfig, get_device_config


def test_device_config():
    """Test device configuration functionality."""
    print("=== Testing Device Configuration ===")

    # Test default initialization
    print("\n1. Testing default device configuration...")
    config = DeviceConfig()
    print(f"   Default device: {config.get_device()}")
    print(f"   PyTorch threads: {torch.get_num_threads()}")

    # Test CPU device
    print("\n2. Testing CPU device...")
    cpu_config = DeviceConfig("cpu")
    print(f"   CPU device: {cpu_config.get_device()}")

    # Test CUDA device (if available)
    if torch.cuda.is_available():
        print("\n3. Testing CUDA device...")
        cuda_config = DeviceConfig("cuda")
        print(f"   CUDA device: {cuda_config.get_device()}")
        print(f"   PyTorch threads: {torch.get_num_threads()}")
    else:
        print("\n3. CUDA not available - testing fallback...")
        cuda_config = DeviceConfig("cuda")
        print(f"   Fallback device: {cuda_config.get_device()}")

    # Test invalid device
    print("\n4. Testing invalid device...")
    invalid_config = DeviceConfig("invalid_device")
    print(f"   Fallback device: {invalid_config.get_device()}")

    # Test device info
    print("\n5. Device information:")
    device_info = config.get_device_info()
    for key, value in device_info.items():
        print(f"   {key}: {value}")

    print("\n=== Device Configuration Tests Completed ===")


def test_tensor_operations():
    """Test tensor operations on configured device."""
    print("\n=== Testing Tensor Operations ===")

    config = get_device_config()
    device = config.get_device()

    # Create test tensors
    x = torch.randn(100, 50)
    y = torch.randn(50, 25)

    # Move to device
    x = config.to_device(x)
    y = config.to_device(y)

    print(f"   Tensor x device: {x.device}")
    print(f"   Tensor y device: {y.device}")

    # Perform operation
    z = torch.mm(x, y)
    print(f"   Result tensor device: {z.device}")
    print(f"   Operation successful: {z.shape}")


if __name__ == "__main__":
    test_device_config()
    test_tensor_operations()
    print("\n=== All Tests Completed Successfully ===")
