#!/usr/bin/env python3
"""
Install required dependencies for the English Assistant
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"   Error: {e.stderr.strip()}")
        return False

def main():
    print("🚀 Setting up English Assistant dependencies...")
    print("=" * 50)
    
    # Install Python packages
    packages = [
        "spacy>=3.7.0",
        "transformers>=4.30.0", 
        "torch>=2.0.0",
        "requests>=2.31.0"
    ]
    
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️  Failed to install {package}, continuing...")
    
    # Download spaCy model
    if not run_command("python -m spacy download en_core_web_sm", "Downloading spaCy English model"):
        print("⚠️  Failed to download spaCy model")
        print("   You can try manually: python -m spacy download en_core_web_sm")
    
    print("\n" + "=" * 50)
    print("🎉 Setup complete!")
    print("\n📋 Next steps:")
    print("1. Set HUGGINGFACE_API_TOKEN environment variable (optional)")
    print("2. Run: python scripts/spacy_server.py")
    print("3. Start your Next.js application")
    print("\n🔗 Get Hugging Face token at: https://huggingface.co/settings/tokens")

if __name__ == '__main__':
    main()
