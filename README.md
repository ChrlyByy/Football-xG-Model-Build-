# Football Expected Goals (xG) Predictor

A logistic regression model built in Python to quantify the quality of football scoring chances. This project demonstrates the full data science pipeline—from data acquisition and cleaning to feature engineering, model training, and evaluation—to predict the probability that a shot results in a goal.

## Model Performance

The trained model achieves the following performance metrics on the test set:

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Log Loss** | 0.282 | The model provides well-calibrated probability estimates. A significant improvement over the baseline of ~0.69 (random guessing). |
| **ROC-AUC** | 0.829 | The model has strong discriminatory power. Given a random goal and a random non-goal, there's an 83% chance the model assigns higher xG to the actual goal. |
| **Brier Score** | 0.082 | The predicted probabilities are close to real-world outcomes, indicating good calibration. |

**Key Insight:** These results indicate a production-ready model that reliably ranks scoring chances and outputs meaningful xG values that reflect actual goal probabilities.

## Methodology

### 1. Data Preparation
- **Source:** StatsBomb open event data
- **Cleaning:** Removed own goals (defensive errors), handled missing values, corrected data inconsistencies (e.g., corner kicks properly flagged as set pieces)
- **Feature Engineering:**
  - `distance_to_goal`: Euclidean distance from shot location to goal center
  - `angle_to_goal`: Angular width of goal from shot location
  - **Contextual features:** `BigChance`, `Penalty`, `Head`, `FastBreak`, `KeyPass`

### 2. Modeling Approach
- **Algorithm:** Logistic Regression with L1 regularization (`penalty='l1'`)
- **Rationale:** Logistic regression naturally outputs probabilities between 0 and 1, making it ideal for xG prediction
- **Hyperparameters:** `C=100`, `max_iter=100`, `solver='liblinear'`
- **Train/Test Split:** 80/20 stratified split to preserve class distribution

### 3. Evaluation Strategy
Used three complementary metrics suitable for probability models:
- **Log Loss:** Primary metric measuring quality of probability estimates
- **ROC-AUC:** Measures ranking capability (goal vs. non-goal discrimination)
- **Brier Score:** Assesses calibration of predicted probabilities

## Getting Started

### Prerequisites
- Python 3.8+
- Git

### Installation
```bash
git clone https://github.com/ChrlyByy/Football-xG-Model-Build-.git
cd Football-xG-Model-Build-
pip install -r requirements.txt
