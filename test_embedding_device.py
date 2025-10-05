"""
Test embedding model with device configuration.
"""

import torch
from device_config import initialize_device_config
from embedding_model import create_industrial_embedding_model


def test_embedding_with_device():
    """Test embedding model with different device configurations."""
    print("=== Testing Embedding Model with Device Configuration ===")
    
    # Test with CPU
    print("\n1. Testing with CPU device...")
    try:
        cpu_config = initialize_device_config('cpu')
        print(f"   Device configured: {cpu_config.get_device()}")
        
        model = create_industrial_embedding_model(
            model_tier="standard",
            device="cpu",
            enable_adaptive_caching=False  # Simplified for testing
        )
        
        # Test encoding
        test_texts = [
            "This is a test sentence for CPU processing.",
            "Another sentence to verify CPU functionality."
        ]
        
        embeddings = model.encode(test_texts)
        print(f"   ✓ CPU encoding successful: {embeddings.shape}")
        
        # Get model diagnostics
        diagnostics = model.get_comprehensive_diagnostics()
        print(f"   ✓ Model loaded: {diagnostics['system_status']['model_loaded']}")
        print(f"   ✓ Model name: {diagnostics['model_info']['name']}")
        print(f"   ✓ Model device: {model.device}")
        
    except Exception as e:
        print(f"   ✗ CPU test failed: {e}")
    
    # Test with CUDA (will fallback to CPU if not available)
    print("\n2. Testing with CUDA device (may fallback to CPU)...")
    try:
        cuda_config = initialize_device_config('cuda')
        print(f"   Device configured: {cuda_config.get_device()}")
        print(f"   PyTorch threads: {torch.get_num_threads()}")
        
        model_cuda = create_industrial_embedding_model(
            model_tier="standard",
            device="cuda",
            enable_adaptive_caching=False  # Simplified for testing
        )
        
        # Test encoding
        embeddings_cuda = model_cuda.encode(test_texts)
        print(f"   ✓ CUDA/CPU encoding successful: {embeddings_cuda.shape}")
        print(f"   ✓ Model device: {model_cuda.device}")
        
    except Exception as e:
        print(f"   ✗ CUDA test failed: {e}")
    
    print("\n=== Embedding Device Tests Completed ===")


if __name__ == "__main__":
    test_embedding_with_device()