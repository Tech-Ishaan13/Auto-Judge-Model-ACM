#!/usr/bin/env python3
"""
Train both classification and regression models on the SAME data split
Then test hybrid approach properly
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore')

def preprocess_text(text):
    """Clean and normalize text"""
    text = re.sub(r'\$[^$]*\$', ' MATHFORMULA ', text)
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', ' LATEXCMD ', text)
    text = re.sub(r'\b\d{4,}\b', ' LARGENUM ', text)
    text = re.sub(r'\b\d+\b', ' NUM ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def extract_features(row):
    """Extract engineered features"""
    full_text = row['full_text']
    description = row['description']
    title = row['title']
    input_desc = row['input_description']
    
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

def prepare_data():
    """Load and prepare data with single split"""
    print("Loading and preparing data...")
    
    df = pd.read_json("problems_data.jsonl", lines=True)
    print(f"Dataset: {len(df)} problems")
    
    # Create full text
    df["full_text"] = (
        df["title"] + " " +
        df["description"] + " " +
        df["input_description"] + " " +
        df["output_description"]
    ).str.lower()
    
    df["processed_text"] = df["full_text"].apply(preprocess_text)
    
    # Single split for both models (no stratification to keep it simple)
    X_text = df["processed_text"]
    y_class = df["problem_class"]
    y_score = df["problem_score"]
    
    X_train_text, X_test_text, y_train_class, y_test_class, y_train_score, y_test_score = train_test_split(
        X_text, y_class, y_score, test_size=0.2, random_state=42
    )
    
    train_indices = X_train_text.index
    test_indices = X_test_text.index
    
    df_train = df.loc[train_indices]
    df_test = df.loc[test_indices]
    
    print(f"Train set: {len(df_train)} samples")
    print(f"Test set: {len(df_test)} samples")
    
    return df_train, df_test, X_train_text, X_test_text, y_train_class, y_test_class, y_train_score, y_test_score

def train_classification_model(df_train, X_train_text, y_train_class):
    """Train classification model"""
    print("\n=== TRAINING CLASSIFICATION MODEL ===")
    
    # TF-IDF features
    tfidf_word = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=3,
        max_df=0.85,
        sublinear_tf=True
    )
    
    tfidf_char = TfidfVectorizer(
        max_features=3000,
        ngram_range=(3, 4),
        analyzer='char',
        min_df=5,
        max_df=0.9
    )
    
    X_train_tfidf_word = tfidf_word.fit_transform(X_train_text)
    X_train_tfidf_char = tfidf_char.fit_transform(X_train_text)
    
    # Engineered features
    X_train_engineered = np.array([
        extract_features(row) for _, row in df_train.iterrows()
    ])
    
    scaler = StandardScaler()
    X_train_engineered_scaled = scaler.fit_transform(X_train_engineered)
    
    # Combine features
    X_train_combined = hstack([
        X_train_tfidf_word,
        X_train_tfidf_char,
        X_train_engineered_scaled
    ])
    
    print(f"Classification feature matrix shape: {X_train_combined.shape}")
    
    # Train models
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            max_features='sqrt',
            random_state=42
        ),
        'LogisticRegression': LogisticRegression(
            C=1.5,
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            solver='liblinear'
        )
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_combined, y_train_class)
        trained_models[name] = model
    
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('rf', trained_models['RandomForest']),
            ('gb', trained_models['GradientBoosting']),
            ('lr', trained_models['LogisticRegression'])
        ],
        voting='soft'
    )
    
    ensemble.fit(X_train_combined, y_train_class)
    
    return ensemble, tfidf_word, tfidf_char, scaler

def train_regression_model(df_train, X_train_text, y_train_score):
    """Train regression model"""
    print("\n=== TRAINING REGRESSION MODEL ===")
    
    # Same TF-IDF settings as classification
    tfidf_word = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=3,
        max_df=0.85,
        sublinear_tf=True
    )
    
    tfidf_char = TfidfVectorizer(
        max_features=3000,
        ngram_range=(3, 4),
        analyzer='char',
        min_df=5,
        max_df=0.9
    )
    
    X_train_tfidf_word = tfidf_word.fit_transform(X_train_text)
    X_train_tfidf_char = tfidf_char.fit_transform(X_train_text)
    
    # Same engineered features as classification
    X_train_engineered = np.array([
        extract_features(row) for _, row in df_train.iterrows()
    ])
    
    scaler = StandardScaler()
    X_train_engineered_scaled = scaler.fit_transform(X_train_engineered)
    
    # Combine features
    X_train_combined = hstack([
        X_train_tfidf_word,
        X_train_tfidf_char,
        X_train_engineered_scaled
    ])
    
    print(f"Regression feature matrix shape: {X_train_combined.shape}")
    
    # Train models
    models = {
        'RandomForest': RandomForestRegressor(
            n_estimators=300,
            max_depth=25,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=8,
            subsample=0.8,
            max_features='sqrt',
            random_state=42
        ),
        'Ridge': Ridge(
            alpha=0.5,
            random_state=42
        )
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_combined, y_train_score)
        trained_models[name] = model
    
    # Create ensemble
    ensemble = VotingRegressor([
        (name, model) for name, model in trained_models.items()
    ])
    
    ensemble.fit(X_train_combined, y_train_score)
    
    return ensemble, tfidf_word, tfidf_char, scaler

def test_models(df_test, X_test_text, y_test_class, y_test_score, 
                class_model, class_tfidf_word, class_tfidf_char, class_scaler,
                reg_model, reg_tfidf_word, reg_tfidf_char, reg_scaler):
    """Test both models on the same test set"""
    print("\n=== TESTING MODELS ===")
    
    # Extract features for test set
    X_test_engineered = np.array([
        extract_features(row) for _, row in df_test.iterrows()
    ])
    
    # Test classification model
    X_test_tfidf_word_class = class_tfidf_word.transform(X_test_text)
    X_test_tfidf_char_class = class_tfidf_char.transform(X_test_text)
    X_test_engineered_scaled_class = class_scaler.transform(X_test_engineered)
    X_test_combined_class = hstack([
        X_test_tfidf_word_class,
        X_test_tfidf_char_class,
        X_test_engineered_scaled_class
    ])
    
    class_predictions = class_model.predict(X_test_combined_class)
    class_accuracy = accuracy_score(y_test_class, class_predictions)
    
    # Test regression model
    X_test_tfidf_word_reg = reg_tfidf_word.transform(X_test_text)
    X_test_tfidf_char_reg = reg_tfidf_char.transform(X_test_text)
    X_test_engineered_scaled_reg = reg_scaler.transform(X_test_engineered)
    X_test_combined_reg = hstack([
        X_test_tfidf_word_reg,
        X_test_tfidf_char_reg,
        X_test_engineered_scaled_reg
    ])
    
    reg_predictions = reg_model.predict(X_test_combined_reg)
    reg_mae = mean_absolute_error(y_test_score, reg_predictions)
    reg_rmse = np.sqrt(mean_squared_error(y_test_score, reg_predictions))
    
    print(f"Classification accuracy: {class_accuracy:.1%}")
    print(f"Regression MAE: {reg_mae:.4f}")
    print(f"Regression RMSE: {reg_rmse:.4f}")
    
    return class_predictions, reg_predictions, class_accuracy, reg_mae

def test_hybrid_approach(y_test_score, class_predictions, reg_predictions, class_accuracy, reg_mae):
    """Test hybrid approach: classification constrains regression"""
    print("\n=== TESTING HYBRID APPROACH ===")
    
    class_boundaries = {
        'easy': (1.1, 2.8),
        'medium': (2.8, 5.5),
        'hard': (5.5, 9.7)
    }
    
    adjusted_scores = []
    adjustments_made = 0
    
    for i, (pred_class, pred_score) in enumerate(zip(class_predictions, reg_predictions)):
        min_score, max_score = class_boundaries[pred_class]
        
        if pred_score < min_score:
            adjusted_score = min_score
            adjustments_made += 1
        elif pred_score > max_score:
            adjusted_score = max_score
            adjustments_made += 1
        else:
            adjusted_score = pred_score
        
        adjusted_scores.append(adjusted_score)
    
    adjusted_scores = np.array(adjusted_scores)
    hybrid_mae = mean_absolute_error(y_test_score, adjusted_scores)
    
    print(f"Adjustments made: {adjustments_made}/{len(class_predictions)} ({adjustments_made/len(class_predictions):.1%})")
    print(f"Hybrid MAE: {hybrid_mae:.4f}")
    
    # Analyze impact
    helped = 0
    hurt = 0
    no_change = 0
    
    for i in range(len(y_test_score)):
        true_score = y_test_score.iloc[i]
        orig_error = abs(true_score - reg_predictions[i])
        hybrid_error = abs(true_score - adjusted_scores[i])
        
        if hybrid_error < orig_error:
            helped += 1
        elif hybrid_error > orig_error:
            hurt += 1
        else:
            no_change += 1
    
    print(f"\nConstraint impact:")
    print(f"  Helped: {helped} predictions ({helped/len(y_test_score):.1%})")
    print(f"  Hurt: {hurt} predictions ({hurt/len(y_test_score):.1%})")
    print(f"  No change: {no_change} predictions ({no_change/len(y_test_score):.1%})")
    
    net_benefit = helped - hurt
    print(f"  Net benefit: {net_benefit:+d} predictions ({net_benefit/len(y_test_score):+.1%})")
    
    improvement = reg_mae - hybrid_mae
    print(f"\n=== FINAL RESULT ===")
    print(f"Classification accuracy: {class_accuracy:.1%}")
    print(f"Regression MAE: {reg_mae:.4f}")
    print(f"Hybrid MAE: {hybrid_mae:.4f}")
    print(f"Improvement: {improvement:+.4f} ({improvement/reg_mae:+.1%})")
    
    if hybrid_mae < reg_mae:
        print("✅ SUCCESS: Hybrid approach improves MAE!")
    else:
        print("❌ FAILURE: Hybrid approach does not improve MAE")
    
    return hybrid_mae, improvement

def main():
    """Main function"""
    print("=== TRAINING BOTH MODELS ON SAME SPLIT ===")
    
    # Prepare data with single split
    df_train, df_test, X_train_text, X_test_text, y_train_class, y_test_class, y_train_score, y_test_score = prepare_data()
    
    # Train both models
    class_model, class_tfidf_word, class_tfidf_char, class_scaler = train_classification_model(
        df_train, X_train_text, y_train_class
    )
    
    reg_model, reg_tfidf_word, reg_tfidf_char, reg_scaler = train_regression_model(
        df_train, X_train_text, y_train_score
    )
    
    # Test both models on same test set
    class_predictions, reg_predictions, class_accuracy, reg_mae = test_models(
        df_test, X_test_text, y_test_class, y_test_score,
        class_model, class_tfidf_word, class_tfidf_char, class_scaler,
        reg_model, reg_tfidf_word, reg_tfidf_char, reg_scaler
    )
    
    # Test hybrid approach
    hybrid_mae, improvement = test_hybrid_approach(
        y_test_score, class_predictions, reg_predictions, class_accuracy, reg_mae
    )
    
    # Save models
    print("\nSaving models...")
    os.makedirs("models_same_split", exist_ok=True)
    
    joblib.dump(class_model, "models_same_split/classification.pkl")
    joblib.dump(class_tfidf_word, "models_same_split/class_tfidf_word.pkl")
    joblib.dump(class_tfidf_char, "models_same_split/class_tfidf_char.pkl")
    joblib.dump(class_scaler, "models_same_split/class_scaler.pkl")
    
    joblib.dump(reg_model, "models_same_split/regression.pkl")
    joblib.dump(reg_tfidf_word, "models_same_split/reg_tfidf_word.pkl")
    joblib.dump(reg_tfidf_char, "models_same_split/reg_tfidf_char.pkl")
    joblib.dump(reg_scaler, "models_same_split/reg_scaler.pkl")
    
    print("Models saved to models_same_split/")

if __name__ == "__main__":
    main()