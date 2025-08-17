#!/usr/bin/env python3
"""
Startup script for English Assistant servers
"""

import subprocess
import sys
import time
import requests
import threading
from pathlib import Path

def check_port(port, service_name):
    """Check if a service is running on the specified port"""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=2)
        if response.status_code == 200:
            print(f"✅ {service_name} is running on port {port}")
            return True
    except:
        pass
    return False

def start_spacy_server():
    """Start the spaCy NLP server"""
    print("🚀 Starting spaCy NLP Server...")
    try:
        # Change to scripts directory
        script_dir = Path(__file__).parent
        subprocess.Popen([
            sys.executable, "spacy_server.py"
        ], cwd=script_dir)
        
        # Wait for server to start
        for i in range(30):  # Wait up to 30 seconds
            if check_port(8000, "spaCy Server"):
                return True
            time.sleep(1)
            if i % 5 == 0:
                print(f"   Waiting for spaCy server... ({i+1}/30)")
        
        print("❌ spaCy server failed to start within 30 seconds")
        return False
        
    except Exception as e:
        print(f"❌ Failed to start spaCy server: {e}")
        return False

def main():
    print("🤖 English Assistant - Server Startup")
    print("=" * 50)
    
    # Check if spaCy server is already running
    if check_port(8000, "spaCy Server"):
        print("spaCy server is already running!")
    else:
        if not start_spacy_server():
            print("\n❌ Failed to start required services")
            print("\n🔧 Troubleshooting:")
            print("1. Make sure you've run: python scripts/install_dependencies.py")
            print("2. Check that spaCy model is installed: python -m spacy download en_core_web_sm")
            print("3. Verify Flask is installed: pip install flask flask-cors")
            return False
    
    print("\n" + "=" * 50)
    print("🎉 All servers are running!")
    print("\n📋 Service Status:")
    print("• spaCy NLP Server: http://localhost:8000")
    print("• Next.js App: http://localhost:3000 (start separately)")
    
    print("\n🚀 Next Steps:")
    print("1. Start your Next.js application: npm run dev")
    print("2. Open http://localhost:3000 in your browser")
    print("3. Optional: Set HUGGINGFACE_API_TOKEN for enhanced responses")
    
    print("\n⚠️  Keep this terminal open to maintain the servers")
    print("Press Ctrl+C to stop all servers")
    
    try:
        # Keep the script running
        while True:
            time.sleep(10)
            # Periodic health checks
            if not check_port(8000, "spaCy Server"):
                print("⚠️  spaCy server appears to have stopped")
                break
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down servers...")
        print("Goodbye!")

if __name__ == '__main__':
    main()
