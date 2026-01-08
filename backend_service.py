#!/usr/bin/env python3
"""
Backend service for AutoJudge - integrates with React frontend
Uses the hybrid model (classification + regression with constraints)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

class HybridProblemPredictor:
    """Hybrid predictor using classification to constrain regression predictions"""
    
    def __init__(self):
        self.classification_model = None
        self.regression_model = None
        self.class_tfidf_word = None
        self.class_tfidf_char = None
        self.class_scaler = None
        self.reg_tfidf_word = None
        self.reg_tfidf_char = None
        self.reg_scaler = None
        self.models_loaded = False
    
    def preprocess_text(self, text):
        """Preprocessing for both models (same preprocessing)"""
        text = re.sub(r'\$[^$]*\$', ' MATHFORMULA ', text)
        text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', ' LATEXCMD ', text)
        text = re.sub(r'\b\d{4,}\b', ' LARGENUM ', text)
        text = re.sub(r'\b\d+\b', ' NUM ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()
    
    def extract_features(self, full_text, description, title, input_desc):
        """Extract engineered features (same for both models)"""
        features = []
        
        # Basic length and structure features
        features.extend([
            len(description), len(full_text), len(title),
            len(description.split()), len(input_desc.split()),
            len(re.findall(r'[.!?]', description)), description.count('\n'),
        ])
        
        # Mathematical complexity indicators
        math_patterns = [
            r'[+\-*/%^=<>≤≥∑∏]', r'\$[^$]*\$', r'\\[a-zA-Z]+',
            r'\blog\s*n\b', r'\bn\^?\d+\b'
        ]
        for pattern in math_patterns:
            features.append(len(re.findall(pattern, full_text)))
        
        # Constraint and number analysis
        numbers = re.findall(r'\d+', full_text)
        if numbers:
            nums = [int(x) for x in numbers if len(x) <= 8]
            if nums:
                features.extend([
                    len(nums), len([n for n in nums if n >= 1000]),
                    len([n for n in nums if n >= 100000]),
                    int(any(n >= 10**6 for n in nums)),
                ])
            else:
                features.extend([0, 0, 0, 0])
        else:
            features.extend([0, 0, 0, 0])
        
        # Algorithm keywords
        algo_categories = {
            'graph_tree': ['graph', 'tree', 'node', 'edge', 'vertex', 'dfs', 'bfs', 'path'],
            'dynamic_programming': ['dp', 'dynamic', 'memoization', 'optimal', 'subproblem'],
            'greedy': ['greedy', 'minimum', 'maximum', 'best', 'optimal'],
            'sorting': ['sort', 'sorted', 'order', 'arrange'],
            'searching': ['search', 'find', 'binary', 'locate'],
            'data_structures': ['array', 'list', 'stack', 'queue', 'heap', 'priority'],
            'string_algorithms': ['string', 'substring', 'pattern', 'match', 'palindrome'],
            'number_theory': ['modular', 'modulo', 'gcd', 'prime', 'factor'],
        }
        
        for category, keywords in algo_categories.items():
            count = sum(full_text.count(keyword) for keyword in keywords)
            features.append(count)
        
        # Complexity indicators
        complexity_indicators = [
            'complex', 'complicated', 'difficult', 'advanced', 'sophisticated',
            'simple', 'basic', 'straightforward', 'easy', 'trivial'
        ]
        for word in complexity_indicators:
            features.append(full_text.count(word))
        
        # Input format complexity
        features.extend([
            len(re.findall(r'\b[nmqktij]\b', input_desc.lower())),
            input_desc.lower().count('test case'),
            int('multiple' in input_desc.lower()),
            int('array' in input_desc.lower() or 'list' in input_desc.lower()),
        ])
        
        return features
    
    def load_models(self):
        """Load hybrid models (classification + regression)"""
        try:
            # Load original hybrid models
            self.classification_model = joblib.load("models_same_split/classification.pkl")
            self.class_tfidf_word = joblib.load("models_same_split/class_tfidf_word.pkl")
            self.class_tfidf_char = joblib.load("models_same_split/class_tfidf_char.pkl")
            self.class_scaler = joblib.load("models_same_split/class_scaler.pkl")
            
            # Load original regression model
            self.regression_model = joblib.load("models_same_split/regression.pkl")
            self.reg_tfidf_word = joblib.load("models_same_split/reg_tfidf_word.pkl")
            self.reg_tfidf_char = joblib.load("models_same_split/reg_tfidf_char.pkl")
            self.reg_scaler = joblib.load("models_same_split/reg_scaler.pkl")
            
            self.models_loaded = True
            return True
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False
    
    def predict_hybrid(self, description, input_format, output_format):
        """Predict using hybrid approach (classification constrains regression)"""
        if not self.models_loaded:
            return None
        
        # Prepare text (use title as empty since frontend doesn't have it)
        title = ""
        full_text = f"{title} {description} {input_format} {output_format}".lower()
        processed_text = self.preprocess_text(full_text)
        
        # Extract engineered features
        engineered_features = np.array([self.extract_features(
            full_text, description, title, input_format
        )])
        
        # Classification prediction
        tfidf_word_class = self.class_tfidf_word.transform([processed_text])
        tfidf_char_class = self.class_tfidf_char.transform([processed_text])
        engineered_scaled_class = self.class_scaler.transform(engineered_features)
        features_combined_class = hstack([tfidf_word_class, tfidf_char_class, engineered_scaled_class])
        
        class_prediction = self.classification_model.predict(features_combined_class)[0]
        class_probabilities = self.classification_model.predict_proba(features_combined_class)[0]
        classes = self.classification_model.classes_
        prob_dict = {classes[i]: class_probabilities[i] for i in range(len(classes))}
        
        # Regression prediction
        tfidf_word_reg = self.reg_tfidf_word.transform([processed_text])
        tfidf_char_reg = self.reg_tfidf_char.transform([processed_text])
        engineered_scaled_reg = self.reg_scaler.transform(engineered_features)
        features_combined_reg = hstack([tfidf_word_reg, tfidf_char_reg, engineered_scaled_reg])
        
        score_prediction = self.regression_model.predict(features_combined_reg)[0]
        
        # Apply class constraints (hybrid approach)
        class_boundaries = {
            'easy': (1.1, 2.8),
            'medium': (2.8, 5.5),
            'hard': (5.5, 9.7)
        }
        
        min_score, max_score = class_boundaries[class_prediction]
        
        if score_prediction < min_score:
            constrained_score = min_score
            constraint_applied = True
        elif score_prediction > max_score:
            constrained_score = max_score
            constraint_applied = True
        else:
            constrained_score = score_prediction
            constraint_applied = False
        
        return {
            'class': class_prediction,
            'probabilities': prob_dict,
            'raw_score': float(score_prediction),
            'constrained_score': float(constrained_score),
            'constraint_applied': constraint_applied
        }

# Initialize predictor
predictor = HybridProblemPredictor()
models_loaded = predictor.load_models()

def generate_explanation(class_pred, score, full_text):
    """Generate explanation based on predictions and text analysis"""
    
    # Analyze text features
    math_symbols = len(re.findall(r'[+\-*/%^=<>≤≥∑∏]', full_text))
    algo_keywords = ['graph', 'tree', 'dp', 'dynamic', 'greedy', 'sort', 'search', 'binary']
    algo_count = sum(full_text.lower().count(keyword) for keyword in algo_keywords)
    
    # Base explanations by class
    explanations = {
        'easy': [
            "This scroll reveals a straightforward algorithmic challenge, suitable for apprentice coders.",
            "The mystical patterns suggest a direct approach with minimal complexity.",
            "A gentle introduction to the arcane arts of programming logic."
        ],
        'medium': [
            "This manuscript demands intermediate mastery of algorithmic principles.",
            "The complexity weaves through multiple layers, requiring careful consideration.",
            "A balanced challenge that tests both logic and implementation skills."
        ],
        'hard': [
            "This ancient text conceals profound algorithmic depths, challenging even seasoned practitioners.",
            "The intricate patterns demand advanced techniques and deep understanding.",
            "A formidable trial that separates novices from true algorithm masters."
        ]
    }
    
    base_explanation = explanations.get(class_pred, explanations['medium'])[0]
    
    # Add specific insights based on analysis
    insights = []
    
    if math_symbols > 10:
        insights.append("Heavy mathematical computation suggests O(n²) or higher complexity.")
    elif math_symbols > 3:
        insights.append("Moderate mathematical operations indicate O(n log n) complexity.")
    
    if algo_count > 5:
        insights.append("Multiple algorithmic concepts interweave throughout this challenge.")
    elif algo_count > 2:
        insights.append("Classic algorithmic patterns emerge from the textual analysis.")
    
    if score > 7:
        insights.append("The mystic potency rating suggests expert-level optimization requirements.")
    elif score > 5:
        insights.append("Intermediate algorithmic techniques will be essential for success.")
    
    # Combine explanation
    if insights:
        return f"{base_explanation} {' '.join(insights)}"
    else:
        return base_explanation

def generate_tags(class_pred, score, full_text):
    """Generate relevant tags based on predictions and text analysis"""
    tags = []
    
    # Algorithm-based tags
    algo_keywords = {
        'dynamic programming': ['dp', 'dynamic', 'memoization'],
        'graph theory': ['graph', 'tree', 'node', 'edge', 'dfs', 'bfs'],
        'greedy': ['greedy', 'optimal'],
        'sorting': ['sort', 'sorted'],
        'binary search': ['binary', 'search'],
        'data structures': ['array', 'list', 'stack', 'queue'],
        'string algorithms': ['string', 'substring', 'pattern'],
        'number theory': ['modular', 'gcd', 'prime']
    }
    
    for tag, keywords in algo_keywords.items():
        if any(keyword in full_text.lower() for keyword in keywords):
            tags.append(tag)
    
    # Difficulty-based tags
    if class_pred == 'easy':
        tags.extend(['implementation', 'basic logic'])
    elif class_pred == 'medium':
        tags.extend(['problem solving', 'algorithms'])
    else:
        tags.extend(['advanced algorithms', 'optimization'])
    
    # Score-based tags
    if score > 7:
        tags.append('expert level')
    elif score > 5:
        tags.append('competitive programming')
    else:
        tags.append('beginner friendly')
    
    return tags[:6]  # Limit to 6 tags

@app.route('/predict', methods=['POST'])
def predict_difficulty():
    """API endpoint for difficulty prediction"""
    if not models_loaded:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        data = request.json
        description = data.get('description', '')
        input_format = data.get('inputFormat', '')
        output_format = data.get('outputFormat', '')
        
        if not description.strip():
            return jsonify({'error': 'Description is required'}), 400
        
        # Get hybrid predictions
        result = predictor.predict_hybrid(description, input_format, output_format)
        
        if result is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        # Map class to frontend format
        class_mapping = {
            'easy': 'Easy',
            'medium': 'Medium', 
            'hard': 'Hard'
        }
        
        difficulty_class = class_mapping.get(result['class'], 'Medium')
        
        # Use constrained score from hybrid approach
        hybrid_score = result['constrained_score']
        
        # Convert hybrid score to realistic Codeforces-like rating based on class
        # Easy: 800-1200, Medium: 1200-1600, Hard: 1600-3500
        if result['class'] == 'easy':
            # Easy problems: score 1.1-2.8 -> rating 800-1200
            cf_score = int(800 + (hybrid_score - 1.1) * (1200 - 800) / (2.8 - 1.1))
        elif result['class'] == 'medium':
            # Medium problems: score 2.8-5.5 -> rating 1200-1600
            cf_score = int(1200 + (hybrid_score - 2.8) * (1600 - 1200) / (5.5 - 2.8))
        else:  # hard
            # Hard problems: score 5.5-9.7 -> rating 1600-3500
            cf_score = int(1600 + (hybrid_score - 5.5) * (3500 - 1600) / (9.7 - 5.5))
        
        # Ensure rating stays within bounds
        cf_score = max(800, min(3500, cf_score))
        
        # Round to nearest 100 (800, 900, 1000, 1100, etc.)
        cf_score = round(cf_score / 100) * 100
        
        # Generate explanation and tags using hybrid score
        full_text = f"{description} {input_format} {output_format}"
        explanation = generate_explanation(result['class'], hybrid_score, full_text)
        tags = generate_tags(result['class'], hybrid_score, full_text)
        
        response = {
            'difficultyClass': difficulty_class,
            'difficultyScore': cf_score,
            'explanation': explanation,
            'tags': tags,
            'confidence': {
                'easy': float(result['probabilities'].get('easy', 0)),
                'medium': float(result['probabilities'].get('medium', 0)),
                'hard': float(result['probabilities'].get('hard', 0))
            },
            'rawScore': float(result['raw_score']),
            'hybridScore': float(hybrid_score),
            'constraintApplied': result['constraint_applied']
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': models_loaded
    })

if __name__ == '__main__':
    if not models_loaded:
        print("❌ Failed to load hybrid models. Please ensure model files are in the 'models_same_split/' directory.")
    else:
        print("✅ Hybrid models loaded successfully!")
        print("🚀 Starting AutoJudge backend service with hybrid model...")
        print("📡 Backend running on: http://localhost:5000")
        print("🎯 Using hybrid approach: Classification constrains Regression")
        print("📊 Model metrics: 56.3% classification accuracy, 1.6305 MAE")
    
    app.run(debug=True, host='0.0.0.0', port=5000)