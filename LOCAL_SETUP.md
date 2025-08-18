# 🚀 Running English Assistant Locally

## Quick Start (Recommended)

1. **Clone and install dependencies:**
   \`\`\`bash
   git clone <your-repo-url>
   cd english-assistant
   npm run setup-full
   \`\`\`

2. **Start the application:**
   \`\`\`bash
   # Terminal 1: Start AI servers
   npm run start-ai

   # Terminal 2: Start web interface  
   npm run dev
   \`\`\`

3. **Open your browser:**
   - Web interface: http://localhost:3000
   - AI server health: http://localhost:8000/health

## What's Included

✅ **spaCy NLP Processing** - Text analysis, entity recognition, grammar checking  
✅ **Voice Input/Output** - Speech-to-text and text-to-speech  
✅ **Learning Modes** - Casual, focused, and advanced learning  
✅ **Progress Tracking** - Vocabulary, grammar, and fluency metrics  
✅ **Smart Suggestions** - Writing improvements and learning points  
🔧 **LLM Integration** - Enhanced responses (requires API token)

## Manual Setup (If needed)

### Prerequisites
- Node.js 18+
- Python 3.8+
- pip

### Step-by-step
\`\`\`bash
# 1. Install Node.js dependencies
npm install

# 2. Install Python dependencies
npm run setup
# Or manually: python3 scripts/install_dependencies.py

# 3. Start servers
npm run start-ai    # AI servers
npm run dev         # Web interface
\`\`\`

## Optional: Enhanced LLM Responses

For better AI responses, add your Hugging Face token:

\`\`\`bash
# Option 1: Environment variable
export HUGGINGFACE_API_TOKEN=your_token_here

# Option 2: Create .env.local file
echo "HUGGINGFACE_API_TOKEN=your_token_here" > .env.local
\`\`\`

Get your token at: https://huggingface.co/settings/tokens

## Troubleshooting

**Common Issues:**
- **spaCy model missing:** Run `python -m spacy download en_core_web_sm`
- **Port conflicts:** AI server uses port 8000, web app uses 3000
- **Python dependencies fail:** Try `pip install --upgrade pip` first
- **Permission errors:** Run `chmod +x scripts/*.py` on Unix systems

**Health Checks:**
- spaCy server: http://localhost:8000/health
- Check logs in terminal for detailed error messages

## Architecture

- **Frontend:** Next.js with React, Tailwind CSS, shadcn/ui
- **Backend:** Python Flask server for NLP processing
- **AI:** spaCy for linguistics, Hugging Face for LLM responses
- **Voice:** Web Speech API for browser-based voice I/O
