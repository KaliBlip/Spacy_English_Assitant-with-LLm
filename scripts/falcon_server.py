#!/usr/bin/env python3
"""
Local Falcon LLM Server
Provides local inference for Hugging Face Falcon models
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import gc
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables for model and tokenizer
model = None
tokenizer = None
generator = None

def load_falcon_model():
    """Load Falcon model and tokenizer"""
    global model, tokenizer, generator
    
    try:
        model_name = "tiiuae/falcon-7b-instruct"
        
        logger.info(f"Loading Falcon model: {model_name}")
        logger.info("This may take several minutes and requires significant GPU memory...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with optimizations
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_8bit=True,  # Use 8-bit quantization to save memory
        )
        
        # Create generation pipeline
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        
        logger.info("Falcon model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load Falcon model: {str(e)}")
        logger.error("Make sure you have sufficient GPU memory (16GB+ recommended)")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_memory": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
    })

@app.route('/generate', methods=['POST'])
def generate_response():
    """Generate response using Falcon model"""
    try:
        if not generator:
            return jsonify({"error": "Falcon model not loaded"}), 500
            
        data = request.get_json()
        prompt = data.get('prompt', '')
        max_tokens = data.get('max_tokens', 200)
        temperature = data.get('temperature', 0.7)
        
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        
        logger.info(f"Generating response for prompt length: {len(prompt)}")
        
        # Generate response
        with torch.no_grad():
            outputs = generator(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Extract generated text
        generated_text = outputs[0]['generated_text']
        
        # Remove the original prompt from the response
        if generated_text.startswith(prompt):
            response = generated_text[len(prompt):].strip()
        else:
            response = generated_text.strip()
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Generated response length: {len(response)}")
        
        return jsonify({
            "response": response,
            "prompt_length": len(prompt),
            "response_length": len(response)
        })
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return jsonify({"error": f"Generation failed: {str(e)}"}), 500

@app.route('/unload', methods=['POST'])
def unload_model():
    """Unload model to free memory"""
    global model, tokenizer, generator
    
    try:
        if model:
            del model
        if tokenizer:
            del tokenizer
        if generator:
            del generator
            
        model = None
        tokenizer = None
        generator = None
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model unloaded successfully")
        return jsonify({"status": "Model unloaded"})
        
    except Exception as e:
        logger.error(f"Error unloading model: {str(e)}")
        return jsonify({"error": f"Unload failed: {str(e)}"}), 500

if __name__ == '__main__':
    print("🚀 Starting Falcon LLM Server...")
    print("=" * 50)
    print("⚠️  WARNING: This server requires significant GPU memory (16GB+ recommended)")
    print("⚠️  Loading Falcon-7B model may take several minutes")
    print("\n📋 System Requirements:")
    print("   - CUDA-compatible GPU with 16GB+ VRAM")
    print("   - PyTorch with CUDA support")
    print("   - transformers, accelerate, bitsandbytes libraries")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\n✅ GPU detected: {gpu_name}")
        print(f"✅ GPU memory: {gpu_memory:.1f} GB")
        
        if gpu_memory < 15:
            print("⚠️  Warning: GPU memory may be insufficient for Falcon-7B")
            print("   Consider using a smaller model or cloud GPU")
    else:
        print("❌ No CUDA GPU detected. This server requires GPU for reasonable performance.")
        print("   CPU inference will be extremely slow and not recommended.")
    
    # Load model on startup
    print(f"\n🔄 Loading Falcon model...")
    if load_falcon_model():
        print("✅ Model loaded successfully!")
        print(f"\n🌐 Server starting on http://localhost:8001")
        app.run(host='0.0.0.0', port=8001, debug=False)
    else:
        print("❌ Failed to load model. Server not started.")
        print("\n💡 Troubleshooting:")
        print("   1. Ensure you have sufficient GPU memory")
        print("   2. Install required packages: pip install torch transformers accelerate bitsandbytes")
        print("   3. Check CUDA installation: python -c 'import torch; print(torch.cuda.is_available())'")
