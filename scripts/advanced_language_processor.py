#!/usr/bin/env python3
"""
Advanced Language Processing Module
Provides sophisticated linguistic analysis and learning features
"""

import spacy
import json
from collections import Counter, defaultdict
import re
from typing import Dict, List, Any, Tuple

class AdvancedLanguageProcessor:
    def __init__(self):
        """Initialize the advanced language processor"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ Advanced Language Processor initialized successfully")
        except OSError:
            print("❌ spaCy model not found. Please install: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def analyze_writing_style(self, text: str) -> Dict[str, Any]:
        """Analyze writing style and provide detailed feedback"""
        if not self.nlp:
            return {"error": "NLP model not loaded"}
        
        doc = self.nlp(text)
        
        # Sentence analysis
        sentences = list(doc.sents)
        sentence_lengths = [len(sent.text.split()) for sent in sentences]
        
        # Word analysis
        words = [token for token in doc if token.is_alpha and not token.is_stop]
        word_lengths = [len(token.text) for token in words]
        
        # Complexity metrics
        avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
        avg_word_length = sum(word_lengths) / len(word_lengths) if word_lengths else 0
        
        # Readability assessment
        readability_score = self._calculate_readability(sentence_lengths, word_lengths)
        
        # Vocabulary diversity
        unique_words = len(set(token.lemma_.lower() for token in words))
        vocabulary_diversity = unique_words / len(words) if words else 0
        
        # Sentence variety
        sentence_types = self._analyze_sentence_types(sentences)
        
        return {
            "style_metrics": {
                "avg_sentence_length": round(avg_sentence_length, 2),
                "avg_word_length": round(avg_word_length, 2),
                "readability_score": round(readability_score, 2),
                "vocabulary_diversity": round(vocabulary_diversity, 2)
            },
            "sentence_analysis": {
                "total_sentences": len(sentences),
                "sentence_lengths": sentence_lengths,
                "sentence_types": sentence_types
            },
            "recommendations": self._generate_style_recommendations(
                avg_sentence_length, avg_word_length, readability_score, vocabulary_diversity
            )
        }
    
    def detect_advanced_grammar_issues(self, text: str) -> List[Dict[str, Any]]:
        """Detect sophisticated grammar and style issues"""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        issues = []
        
        # Check for passive voice
        passive_constructions = self._find_passive_voice(doc)
        for construction in passive_constructions:
            issues.append({
                "type": "passive_voice",
                "message": f"Consider using active voice instead of passive: '{construction['text']}'",
                "start": construction["start"],
                "end": construction["end"],
                "severity": "medium",
                "suggestion": "Rewrite in active voice for more direct communication"
            })
        
        # Check for wordiness
        wordy_phrases = self._find_wordy_phrases(doc)
        for phrase in wordy_phrases:
            issues.append({
                "type": "wordiness",
                "message": f"Consider simplifying: '{phrase['original']}' → '{phrase['suggested']}'",
                "start": phrase["start"],
                "end": phrase["end"],
                "severity": "low",
                "original": phrase["original"],
                "suggested": phrase["suggested"]
            })
        
        # Check for repetitive words
        repetitive_words = self._find_repetitive_words(doc)
        for word_info in repetitive_words:
            issues.append({
                "type": "repetition",
                "message": f"The word '{word_info['word']}' appears {word_info['count']} times. Consider using synonyms.",
                "severity": "low",
                "word": word_info["word"],
                "count": word_info["count"]
            })
        
        # Check for sentence fragments
        fragments = self._find_sentence_fragments(doc)
        for fragment in fragments:
            issues.append({
                "type": "sentence_fragment",
                "message": f"Possible sentence fragment: '{fragment['text']}'",
                "start": fragment["start"],
                "end": fragment["end"],
                "severity": "high"
            })
        
        return issues
    
    def generate_learning_insights(self, text: str, user_level: str = "intermediate") -> Dict[str, Any]:
        """Generate personalized learning insights based on text analysis"""
        if not self.nlp:
            return {"error": "NLP model not loaded"}
        
        doc = self.nlp(text)
        insights = {
            "grammar_concepts": [],
            "vocabulary_insights": [],
            "style_tips": [],
            "practice_suggestions": []
        }
        
        # Grammar concepts found in text
        grammar_concepts = self._identify_grammar_concepts(doc)
        insights["grammar_concepts"] = grammar_concepts
        
        # Vocabulary analysis
        vocab_insights = self._analyze_vocabulary_level(doc, user_level)
        insights["vocabulary_insights"] = vocab_insights
        
        # Style analysis
        style_tips = self._generate_style_tips(doc, user_level)
        insights["style_tips"] = style_tips
        
        # Practice suggestions
        practice_suggestions = self._suggest_practice_exercises(doc, user_level)
        insights["practice_suggestions"] = practice_suggestions
        
        return insights
    
    def _calculate_readability(self, sentence_lengths: List[int], word_lengths: List[int]) -> float:
        """Calculate a simplified readability score"""
        if not sentence_lengths or not word_lengths:
            return 0
        
        avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)
        avg_word_length = sum(word_lengths) / len(word_lengths)
        
        # Simplified readability formula
        readability = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (avg_word_length / 4.7))
        return max(0, min(100, readability))
    
    def _analyze_sentence_types(self, sentences) -> Dict[str, int]:
        """Analyze types of sentences (declarative, interrogative, etc.)"""
        types = {"declarative": 0, "interrogative": 0, "exclamatory": 0, "imperative": 0}
        
        for sent in sentences:
            text = sent.text.strip()
            if text.endswith("?"):
                types["interrogative"] += 1
            elif text.endswith("!"):
                types["exclamatory"] += 1
            elif sent.root.pos_ == "VERB" and sent.root.dep_ == "ROOT":
                # Simple heuristic for imperative
                if sent.root.tag_ in ["VB", "VBP"]:
                    types["imperative"] += 1
                else:
                    types["declarative"] += 1
            else:
                types["declarative"] += 1
        
        return types
    
    def _find_passive_voice(self, doc) -> List[Dict[str, Any]]:
        """Find passive voice constructions"""
        passive_constructions = []
        
        for token in doc:
            if token.dep_ == "auxpass":  # Auxiliary passive
                # Find the main verb
                for child in token.head.children:
                    if child.dep_ == "agent":  # "by" phrase
                        passive_constructions.append({
                            "text": token.head.sent.text,
                            "start": token.head.sent.start_char,
                            "end": token.head.sent.end_char
                        })
                        break
                else:
                    # Passive without agent
                    passive_constructions.append({
                        "text": token.head.sent.text,
                        "start": token.head.sent.start_char,
                        "end": token.head.sent.end_char
                    })
        
        return passive_constructions
    
    def _find_wordy_phrases(self, doc) -> List[Dict[str, Any]]:
        """Find wordy phrases that can be simplified"""
        wordy_patterns = {
            "in order to": "to",
            "due to the fact that": "because",
            "at this point in time": "now",
            "for the purpose of": "for",
            "in the event that": "if",
            "make a decision": "decide",
            "come to a conclusion": "conclude"
        }
        
        wordy_phrases = []
        text = doc.text.lower()
        
        for wordy, simple in wordy_patterns.items():
            if wordy in text:
                # Find position (simplified)
                start = text.find(wordy)
                if start != -1:
                    wordy_phrases.append({
                        "original": wordy,
                        "suggested": simple,
                        "start": start,
                        "end": start + len(wordy)
                    })
        
        return wordy_phrases
    
    def _find_repetitive_words(self, doc) -> List[Dict[str, Any]]:
        """Find words that are used repetitively"""
        word_counts = Counter()
        
        for token in doc:
            if token.is_alpha and not token.is_stop and len(token.text) > 3:
                word_counts[token.lemma_.lower()] += 1
        
        repetitive = []
        for word, count in word_counts.items():
            if count >= 3:  # Appears 3 or more times
                repetitive.append({"word": word, "count": count})
        
        return repetitive
    
    def _find_sentence_fragments(self, doc) -> List[Dict[str, Any]]:
        """Find potential sentence fragments"""
        fragments = []
        
        for sent in doc.sents:
            # Simple heuristic: sentences without a main verb
            has_main_verb = any(token.pos_ == "VERB" and token.dep_ == "ROOT" for token in sent)
            
            if not has_main_verb and len(sent.text.split()) > 2:
                fragments.append({
                    "text": sent.text,
                    "start": sent.start_char,
                    "end": sent.end_char
                })
        
        return fragments
    
    def _identify_grammar_concepts(self, doc) -> List[Dict[str, Any]]:
        """Identify grammar concepts present in the text"""
        concepts = []
        
        # Verb tenses
        verb_tenses = defaultdict(int)
        for token in doc:
            if token.pos_ == "VERB":
                verb_tenses[token.tag_] += 1
        
        if verb_tenses:
            concepts.append({
                "concept": "Verb Tenses",
                "description": "Different verb tenses found in your text",
                "examples": list(verb_tenses.keys())[:3]
            })
        
        # Complex sentences
        complex_sentences = 0
        for sent in doc.sents:
            if len([token for token in sent if token.dep_ in ["advcl", "ccomp", "xcomp"]]) > 0:
                complex_sentences += 1
        
        if complex_sentences > 0:
            concepts.append({
                "concept": "Complex Sentences",
                "description": f"Found {complex_sentences} complex sentences with subordinate clauses",
                "examples": ["Sentences with dependent clauses"]
            })
        
        return concepts
    
    def _analyze_vocabulary_level(self, doc, user_level: str) -> List[Dict[str, Any]]:
        """Analyze vocabulary complexity and provide insights"""
        insights = []
        
        # Word length analysis
        long_words = [token.text for token in doc if token.is_alpha and len(token.text) > 7]
        if long_words:
            insights.append({
                "type": "vocabulary_complexity",
                "message": f"You used {len(long_words)} complex words",
                "examples": long_words[:3],
                "tip": "Complex vocabulary shows advanced language skills"
            })
        
        # Academic/formal words (simplified detection)
        formal_indicators = ["however", "therefore", "furthermore", "consequently", "nevertheless"]
        formal_words = [word for word in formal_indicators if word in doc.text.lower()]
        
        if formal_words:
            insights.append({
                "type": "formal_language",
                "message": "You used formal transitional words",
                "examples": formal_words,
                "tip": "Formal language is great for academic and professional writing"
            })
        
        return insights
    
    def _generate_style_tips(self, doc, user_level: str) -> List[str]:
        """Generate style improvement tips"""
        tips = []
        
        sentences = list(doc.sents)
        if len(sentences) > 1:
            sentence_lengths = [len(sent.text.split()) for sent in sentences]
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            
            if avg_length > 20:
                tips.append("Try varying sentence length for better flow")
            elif avg_length < 8:
                tips.append("Consider combining some short sentences for variety")
        
        # Check for transition words
        transitions = ["however", "therefore", "moreover", "furthermore", "in addition"]
        has_transitions = any(trans in doc.text.lower() for trans in transitions)
        
        if not has_transitions and len(sentences) > 2:
            tips.append("Add transition words to improve flow between sentences")
        
        return tips
    
    def _suggest_practice_exercises(self, doc, user_level: str) -> List[Dict[str, str]]:
        """Suggest relevant practice exercises"""
        suggestions = []
        
        # Based on grammar patterns found
        verb_count = len([token for token in doc if token.pos_ == "VERB"])
        if verb_count > 3:
            suggestions.append({
                "type": "grammar",
                "exercise": "Verb Tense Consistency",
                "description": "Practice maintaining consistent verb tenses throughout your writing"
            })
        
        # Based on sentence complexity
        complex_sentences = sum(1 for sent in doc.sents 
                               if len([token for token in sent if token.dep_ in ["advcl", "ccomp"]]) > 0)
        
        if complex_sentences == 0:
            suggestions.append({
                "type": "structure",
                "exercise": "Complex Sentence Formation",
                "description": "Practice creating sentences with dependent clauses"
            })
        
        return suggestions
    
    def _generate_style_recommendations(self, avg_sent_len: float, avg_word_len: float, 
                                      readability: float, vocab_diversity: float) -> List[str]:
        """Generate style recommendations based on metrics"""
        recommendations = []
        
        if avg_sent_len > 25:
            recommendations.append("Consider breaking long sentences into shorter ones for clarity")
        elif avg_sent_len < 10:
            recommendations.append("Try combining some short sentences for better flow")
        
        if readability < 30:
            recommendations.append("Your text is quite complex. Consider simplifying for broader readability")
        elif readability > 80:
            recommendations.append("Your text is very readable. You might add some complexity for sophistication")
        
        if vocab_diversity < 0.5:
            recommendations.append("Try using more varied vocabulary to make your writing more engaging")
        
        if avg_word_len > 6:
            recommendations.append("You use sophisticated vocabulary. Great for formal writing!")
        
        return recommendations

if __name__ == "__main__":
    processor = AdvancedLanguageProcessor()
    
    # Test text
    test_text = """
    The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.
    However, it is quite simple. We should analyze more complex texts to understand advanced features.
    """
    
    print("🧪 Testing Advanced Language Processor...")
    
    # Test style analysis
    style_analysis = processor.analyze_writing_style(test_text)
    print("\n📊 Style Analysis:")
    print(json.dumps(style_analysis, indent=2))
    
    # Test grammar issues
    grammar_issues = processor.detect_advanced_grammar_issues(test_text)
    print(f"\n🔍 Grammar Issues Found: {len(grammar_issues)}")
    
    # Test learning insights
    insights = processor.generate_learning_insights(test_text)
    print("\n💡 Learning Insights:")
    print(json.dumps(insights, indent=2))
