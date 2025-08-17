#!/usr/bin/env python3
"""
Install requirements for Falcon LLM integration
"""

import subprocess
import sys
import torch

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 Installing Falcon LLM Requirements")
    print("=" * 50)
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        
        if gpu_memory < 15:
            print("⚠️  Warning: GPU memory may be insufficient for Falcon-7B")
    else:
        print("⚠️  No CUDA GPU detected. Falcon inference will be very slow on CPU.")
    
    # Install required packages
    packages = [
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "bitsandbytes>=0.39.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0"
    ]
    
    print(f"\n📦 Installing Falcon-specific packages...")
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️  Failed to install {package}")
    
    # Test Falcon model loading (just tokenizer to avoid memory issues)
    print(f"\n🧪 Testing Falcon model access...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct", trust_remote_code=True)
        print("✅ Falcon tokenizer loaded successfully")
        print("✅ Model access confirmed")
    except Exception as e:
        print(f"❌ Falcon model test failed: {e}")
        print("This might be due to network issues or missing dependencies")
    
    print(f"\n🎉 Falcon requirements installation complete!")
    print(f"\n📝 Next steps:")
    print(f"1. Set HUGGINGFACE_API_TOKEN environment variable for API access")
    print(f"2. For local inference: python scripts/falcon_server.py")
    print(f"3. For API-only usage: The system will automatically use HF API")
    
    print(f"\n⚠️  Important notes:")
    print(f"   - Local Falcon inference requires 16GB+ GPU memory")
    print(f"   - API usage requires Hugging Face account and token")
    print(f"   - System will fallback gracefully if neither is available")

if __name__ == "__main__":
    main()
