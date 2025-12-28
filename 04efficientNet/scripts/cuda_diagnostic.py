import torch
import sys

def diagnose_cuda():
    print("=== CUDA Diagnostic Report ===")
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if not cuda_available:
        print("❌ CUDA is not available. Possible reasons:")
        print("  1. No NVIDIA GPU installed")
        print("  2. CUDA drivers not installed")
        print("  3. PyTorch not installed with CUDA support")
        print("  4. CUDA version mismatch")
        return
    
    # If CUDA is available, get more details
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    if torch.cuda.device_count() > 0:
        print(f"Current GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Test tensor creation on GPU
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.mm(x, y)
            print("✅ Successfully created and operated on tensors on GPU")
        except Exception as e:
            print(f"❌ Failed to create tensors on GPU: {e}")
    else:
        print("❌ No GPUs detected")

if __name__ == "__main__":
    diagnose_cuda()
