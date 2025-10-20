# League of Legends Match Predictor

## Project Description

This project builds a logistic regression model using PyTorch to predict the outcomes of League of Legends matches. The model is trained on match data to classify whether a team will win or lose based on game statistics.

## Dataset

The project uses `league_of_legends_data_large.csv` containing 1,000 match records with 8 features. The data is split into 80% training (800 samples) and 20% testing (200 samples).

## Implementation Steps

### 1. Data Loading and Preprocessing
- Load the dataset from CSV file
- Split features (X) and target variable (win/loss)
- Split data into train and test sets (80/20 split, random_state=42)
- Standardize features using StandardScaler
- Convert data to PyTorch tensors

### 2. Model Architecture
- Logistic regression model with one linear layer
- Input: 8 features
- Output: Binary classification (win/loss)
- Activation: Sigmoid function

### 3. Model Training
- Loss function: Binary Cross-Entropy Loss
- Optimizer: Stochastic Gradient Descent (SGD)
- Learning rate: 0.01
- Epochs: 1,000
- Training includes printing loss every 100 epochs

### 4. Model Optimization
- Apply L2 regularization (weight_decay=0.01)
- Compare performance with and without regularization
- Evaluate on both training and test sets

### 5. Model Evaluation
- Calculate accuracy on train and test sets
- Generate confusion matrix
- Plot ROC curve with AUC score
- Generate classification report with precision, recall, and F1-score

### 6. Model Saving and Loading
- Save trained model to `logistic_regression_model.pth`
- Load model and verify it works correctly
- Test loaded model on test data

### 7. Hyperparameter Tuning
- Test different learning rates: [0.01, 0.05, 0.1]
- Train models with 100 epochs each
- Compare test accuracies and select best learning rate

### 8. Feature Importance
- Extract weights from the trained model
- Rank features by importance (absolute weight values)
- Visualize feature importance with bar plot

## Results

**Model Performance:**
- Training Accuracy: ~54%
- Test Accuracy: ~51%
- AUC Score: 0.50

**Best Hyperparameters:**
- Learning rate: 0.05 (achieved 55% test accuracy)

## Requirements

- Python 3.x
- PyTorch
- pandas
- scikit-learn
- matplotlib
- numpy

## Installation

```bash
pip install torch pandas scikit-learn matplotlib numpy
```

## Usage

1. Download the dataset `league_of_legends_data_large.csv`
2. Open the Jupyter notebook `League_of_Legends_Match_Predictor-1.ipynb`
3. Run all cells in sequence
4. View results and visualizations inline

## Files

- `League_of_Legends_Match_Predictor-1.ipynb` - Main project notebook
- `league_of_legends_data_large.csv` - Dataset (download separately)
- `logistic_regression_model.pth` - Saved trained model
- `README.md` - This file
