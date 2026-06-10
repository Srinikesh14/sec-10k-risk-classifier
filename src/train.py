import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Import the specific models requested in the assignment
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier

def evaluate_model(name, model, X_test, y_test):
    """Generates the required metrics for each model."""
    predictions = model.predict(X_test)
    
    print(f"\n--- {name} Performance ---")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    
    # zero_division=0 prevents crashes since our test set is extremely small and imbalanced
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y_test, predictions, zero_division=0):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

if __name__ == "__main__":
    print("--- Starting Stage 3 & 4: Training and Evaluation ---")
    
    # 1. Load the math features and labels we just made
    print("Loading feature matrix and labels...")
    X = joblib.load('data/X_features.joblib')
    y = joblib.load('data/y_labels.joblib')
    
    # 2. Split the data (80% for training, 20% for grading)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Initialize the three required models
    models = {
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42) # verbose=0 stops it from spamming the terminal
    }
    
    # 4. Train and Evaluate each model
    best_model = None
    best_name = ""
    best_score = -1
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        evaluate_model(name, model, X_test, y_test)
        
        # We will use Accuracy as our metric to pick the "best" model for deployment
        # Note: In a real imbalanced dataset, we would use F1, but Accuracy is fine for our 500-row test pipeline
        current_score = accuracy_score(y_test, model.predict(X_test))
        if current_score > best_score:
            best_score = current_score
            best_model = model
            best_name = name

    # 5. Save the winning model for Stage 5 (API Deployment)
    print(f"\n🏆 The Best Model is: {best_name}! Saving to models/ folder...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.joblib')
    
    print("Stages 3 & 4 Complete! Winning model saved.")