#!/usr/bin/env python3
"""
spaCy NLP Processing Server
Provides text analysis capabilities for the English Assistant
"""

import spacy
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("Successfully loaded spaCy model: en_core_web_sm")
except OSError:
    logger.error("spaCy model 'en_core_web_sm' not found. Please install it with: python -m spacy download en_core_web_sm")
    nlp = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "spacy_loaded": nlp is not None,
        "model": "en_core_web_sm" if nlp else None
    })

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """Analyze text using spaCy NLP pipeline"""
    try:
        if not nlp:
            return jsonify({"error": "spaCy model not loaded"}), 500
            
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Process text with spaCy
        doc = nlp(text)
        
        # Extract entities
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "description": spacy.explain(ent.label_),
                "start": ent.start_char,
                "end": ent.end_char
            })
        
        # Extract tokens with linguistic features
        tokens = []
        for token in doc:
            tokens.append({
                "text": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "tag": token.tag_,
                "dep": token.dep_,
                "is_alpha": token.is_alpha,
                "is_stop": token.is_stop,
                "is_punct": token.is_punct
            })
        
        # Extract sentences
        sentences = []
        for sent in doc.sents:
            sentences.append({
                "text": sent.text,
                "start": sent.start_char,
                "end": sent.end_char
            })
        
        # Basic text statistics
        stats = {
            "num_tokens": len(doc),
            "num_sentences": len(list(doc.sents)),
            "num_entities": len(entities),
            "num_words": len([token for token in doc if token.is_alpha]),
            "num_stop_words": len([token for token in doc if token.is_stop])
        }
        
        # Grammar and style analysis
        grammar_issues = []
        for token in doc:
            # Simple grammar checks
            if token.pos_ == "VERB" and token.tag_ == "VBZ" and token.head.pos_ == "NOUN":
                if token.head.tag_ in ["NNS", "NNPS"]:  # Plural nouns
                    grammar_issues.append({
                        "type": "subject_verb_disagreement",
                        "message": f"Possible subject-verb disagreement: '{token.head.text}' (plural) with '{token.text}' (singular verb)",
                        "start": token.idx,
                        "end": token.idx + len(token.text)
                    })
        
        response = {
            "text": text,
            "entities": entities,
            "tokens": tokens,
            "sentences": sentences,
            "statistics": stats,
            "grammar_issues": grammar_issues,
            "language": doc.lang_
        }
        
        logger.info(f"Analyzed text with {len(entities)} entities and {len(tokens)} tokens")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route('/grammar-check', methods=['POST'])
def grammar_check():
    """Focused grammar checking endpoint"""
    try:
        if not nlp:
            return jsonify({"error": "spaCy model not loaded"}), 500
            
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        doc = nlp(text)
        issues = []
        
        # More comprehensive grammar checks
        for token in doc:
            # Check for common grammar issues
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                # Check subject-verb agreement
                subjects = [child for child in token.children if child.dep_ in ["nsubj", "nsubjpass"]]
                for subj in subjects:
                    if subj.tag_ in ["NNS", "NNPS"] and token.tag_ == "VBZ":
                        issues.append({
                            "type": "subject_verb_agreement",
                            "message": f"Subject '{subj.text}' (plural) doesn't agree with verb '{token.text}' (singular)",
                            "suggestion": f"Use '{token.lemma_}' instead of '{token.text}'",
                            "start": token.idx,
                            "end": token.idx + len(token.text),
                            "severity": "high"
                        })
            
            # Check for double negatives
            if token.dep_ == "neg" and any(child.dep_ == "neg" for child in token.head.children):
                issues.append({
                    "type": "double_negative",
                    "message": "Possible double negative detected",
                    "start": token.idx,
                    "end": token.idx + len(token.text),
                    "severity": "medium"
                })
        
        return jsonify({
            "text": text,
            "issues": issues,
            "issue_count": len(issues)
        })
        
    except Exception as e:
        logger.error(f"Error in grammar check: {str(e)}")
        return jsonify({"error": f"Grammar check failed: {str(e)}"}), 500

if __name__ == '__main__':
    print("Starting spaCy NLP Server...")
    print("Make sure to install required packages:")
    print("pip install spacy flask flask-cors")
    print("python -m spacy download en_core_web_sm")
    print("\nServer will run on http://localhost:8000")
    
    app.run(host='0.0.0.0', port=8000, debug=True)
