# AutoJudge: AI-Powered Programming Problem Difficulty Predictor

An intelligent system that predicts the difficulty of competitive programming problems using a hybrid machine learning approach combining classification and regression models.

## 👤 Author

**Ishaan Arora**

## 📋 Project Overview

AutoJudge analyzes programming problem descriptions and predicts:
1. **Difficulty Class**: Easy, Medium, or Hard
2. **Difficulty Score**: Numerical rating (converted to Codeforces-style 800-3500)

The system uses a **hybrid approach** where classification predictions constrain regression outputs, improving overall prediction accuracy.

## 🎯 Demo Video

[Link to Demo Video] *(Add your 2-3 minute demo video link here)*

## 📊 Dataset

- **Source**: Competitive programming problems (Codeforces-style)
- **Total Samples**: 4,112 problems
- **Train/Test Split**: 80/20 (3,289 train, 823 test)
- **Features**: Problem description, input/output format, constraints
- **Labels**: 
  - Classification: Easy, Medium, Hard
  - Regression: Difficulty score (1.1 - 9.7)

## 🧠 Approach & Models

### Hybrid Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HYBRID MODEL APPROACH                    │
│         (Classification constrains Regression)              │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
┌───────────────────┐                    ┌───────────────────┐
│  CLASSIFICATION   │                    │    REGRESSION     │
│  VotingClassifier │                    │  VotingRegressor  │
├───────────────────┤                    ├───────────────────┤
│ • RandomForest    │                    │ • RandomForest    │
│ • GradientBoost   │                    │ • GradientBoost   │
│ • LogisticReg     │                    │ • Ridge           │
└───────────────────┘                    └───────────────────┘
```

### Feature Engineering (38 Features)

| Category | Count | Description |
|----------|-------|-------------|
| Length & Structure | 7 | Text lengths, word counts, sentences |
| Math Complexity | 5 | Operators, LaTeX, complexity notation |
| Number Analysis | 4 | Constraint sizes, large numbers |
| Algorithm Keywords | 8 | Graph, DP, Greedy, Sort, Search, etc. |
| Complexity Words | 10 | "complex", "simple", "difficult", etc. |
| Input Format | 4 | Variables, test cases, arrays |

### Text Features (TF-IDF)

- **Word-level TF-IDF**: 8,000 features (1-2 ngrams)
- **Character-level TF-IDF**: 3,000 features (3-4 ngrams)

### Hybrid Constraint Logic

```python
Class Boundaries:
├── Easy:   1.1 - 2.8  → Codeforces 800-1200
├── Medium: 2.8 - 5.5  → Codeforces 1200-1600
└── Hard:   5.5 - 9.7  → Codeforces 1600-3500
```

## 📈 Evaluation Metrics

### Classification Results

| Metric | Value |
|--------|-------|
| **Accuracy** | **54.80%** |
| Easy Precision | 0.4722 |
| Medium Precision | 0.4118 |
| Hard Precision | 0.6184 |

### Confusion Matrix

```
                  Predicted
                  Easy    Medium    Hard
Actual Easy        51       42       43
Actual Medium      26       84      152
Actual Hard        31       78      316
```

### Regression Results (Hybrid Model)

| Metric | Value |
|--------|-------|
| **MAE** | **1.6536** |
| **RMSE** | **2.0527** |
| R² Score | 0.1222 |

### Hybrid Improvement

- Adjustments made: 376/823 (45.7%)
- Predictions improved: 208 (25.3%)
- Net benefit: +41 predictions

## 🚀 How to Run Locally

### Prerequisites

```bash
pip install -r requirements.txt
```

### Step 1: Start the Backend Server

```bash
python backend_service.py
```

You should see:
```
✅ Hybrid models loaded successfully!
🚀 Starting AutoJudge backend service with hybrid model...
📡 Backend running on: http://localhost:5000
```

### Step 2: Open the Frontend

Open `simple_frontend.html` in your web browser (double-click or drag into browser).

### Step 3: Make Predictions

1. Enter a problem description
2. (Optional) Add input/output format
3. Click "Cast Prediction Spell"
4. View the predicted difficulty class and score

## 🌐 Web Interface

The web interface provides:

- **Problem Input**: Text areas for description, input format, output format
- **Difficulty Prediction**: Easy/Medium/Hard classification with confidence scores
- **Score Display**: 
  - Raw regression score
  - Hybrid constrained score
  - Codeforces-style rating (800-3500)
- **Explanation**: AI-generated explanation of the prediction
- **Tags**: Relevant algorithm tags based on problem content

### Screenshot

*(Add a screenshot of your web interface here)*

## 📁 Project Structure

```
AutoJudge/
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── problems_data.jsonl            # Dataset
├── train_both_models_same_split.py # Model training code
├── backend_service.py             # Flask API server
├── simple_frontend.html           # Web UI
└── models_same_split/             # Trained models
    ├── classification.pkl
    ├── regression.pkl
    ├── class_tfidf_word.pkl
    ├── class_tfidf_char.pkl
    ├── class_scaler.pkl
    ├── reg_tfidf_word.pkl
    ├── reg_tfidf_char.pkl
    └── reg_scaler.pkl
```

## 🔧 API Endpoints

### POST /predict

Predict difficulty for a programming problem.

**Request:**
```json
{
  "description": "Problem description text",
  "inputFormat": "Input format description",
  "outputFormat": "Output format description"
}
```

**Response:**
```json
{
  "difficultyClass": "Medium",
  "difficultyScore": 1400,
  "confidence": {"easy": 0.2, "medium": 0.5, "hard": 0.3},
  "hybridScore": 4.2,
  "rawScore": 4.5,
  "explanation": "...",
  "tags": ["dynamic programming", "algorithms"]
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true
}
```

## 📚 Dependencies

- Flask & Flask-CORS (Web server)
- scikit-learn (ML models)
- pandas & numpy (Data processing)
- scipy (Sparse matrices)
- joblib (Model serialization)

## 🎓 Acknowledgments

This project was developed as part of a Machine Learning course assignment.
