#!/usr/bin/env python3
"""
Setup script for the English Assistant AI environment
Installs required packages and downloads models
"""

import subprocess
import sys
import os

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
    print("🚀 Setting up English Assistant AI Environment")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} detected")
    
    # Install required packages
    packages = [
        "spacy>=3.7.0",
        "flask>=2.3.0",
        "flask-cors>=4.0.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "requests>=2.31.0"
    ]
    
    print(f"\n📦 Installing Python packages...")
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️  Failed to install {package}, continuing...")
    
    # Download spaCy model
    if not run_command("python -m spacy download en_core_web_sm", "Downloading spaCy English model"):
        print("⚠️  Failed to download spaCy model. You may need to install it manually.")
    
    # Test installations
    print(f"\n🧪 Testing installations...")
    
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy and English model loaded successfully")
    except Exception as e:
        print(f"❌ spaCy test failed: {e}")
    
    try:
        import transformers
        print("✅ Transformers library loaded successfully")
    except Exception as e:
        print(f"❌ Transformers test failed: {e}")
    
    try:
        import flask
        print("✅ Flask loaded successfully")
    except Exception as e:
        print(f"❌ Flask test failed: {e}")
    
    print(f"\n🎉 Setup complete!")
    print(f"\nNext steps:")
    print(f"1. Run the spaCy server: python scripts/spacy_server.py")
    print(f"2. Start the Next.js development server: npm run dev")
    print(f"3. Open http://localhost:3000 in your browser")
    
    print(f"\n📝 Note: For Hugging Face Falcon integration, you'll need:")
    print(f"   - Hugging Face account and API token")
    print(f"   - Additional GPU memory for local model inference")

if __name__ == "__main__":
    main()
