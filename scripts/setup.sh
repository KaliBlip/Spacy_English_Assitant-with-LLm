#!/bin/bash

echo "🤖 English Assistant - Complete Setup"
echo "===================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

echo "✅ Python 3 found"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
python3 scripts/install_dependencies.py

if [ $? -ne 0 ]; then
    echo "❌ Failed to install Python dependencies"
    exit 1
fi

# Check if Node.js is available
if ! command -v npm &> /dev/null; then
    echo "❌ Node.js/npm is required but not installed"
    echo "Please install Node.js from https://nodejs.org/"
    exit 1
fi

echo "✅ Node.js/npm found"

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install

if [ $? -ne 0 ]; then
    echo "❌ Failed to install Node.js dependencies"
    exit 1
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "🚀 To start the English Assistant:"
echo "1. Run: python3 scripts/start_servers.py"
echo "2. In another terminal, run: npm run dev"
echo "3. Open http://localhost:3000 in your browser"
echo ""
echo "💡 Optional: Set HUGGINGFACE_API_TOKEN environment variable for enhanced AI responses"
echo "   Get your token at: https://huggingface.co/settings/tokens"
