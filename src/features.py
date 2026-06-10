import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

def create_target_labels(df):
    """
    Creates a binary target label for classification.
    Rule: If the risk factors mention severe financial distress keywords, 
    we label it as High Risk (1). Otherwise, Normal Risk (0).
    """
    severe_keywords = ['bankruptcy', 'default', 'lawsuit', 'litigation', 'penalty', 'shortage', 'loss', 'debt', 'adverse', 'decline', 'restructuring']
    
    def check_risk(text):
        if not isinstance(text, str):
            return 0
        # If any severe keyword is in the text, flag as 1
        if any(keyword in text for keyword in severe_keywords):
            return 1
        return 0

    df['target_label'] = df['cleaned_text'].apply(check_risk)
    return df

if __name__ == "__main__":
    print("--- Starting Stage 2: Feature Engineering ---")
    
    # 1. Load the clean data from Stage 1
    print("Loading processed data...")
    df = pd.read_csv('data/processed_data.csv')
    
    # Ensure no completely empty rows sneak in
    df['cleaned_text'] = df['cleaned_text'].fillna("")
    
    # 2. Create our classification targets (y)
    print("Generating target labels...")
    df = create_target_labels(df)
    
    # 3. Initialize and fit the TF-IDF Vectorizer (converting text to numbers)
    print("Vectorizing text using TF-IDF...")
    # max_features=1500 limits the math to the top 1500 most important words so it runs fast
    vectorizer = TfidfVectorizer(max_features=1500) 
    
    # X is our mathematical feature matrix
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['target_label'].values
    
    print(f"Created feature matrix with shape: {X.shape}")
    print(f"High Risk companies found: {sum(y)}")
    
    # 4. Save everything for Stage 3 (Training)
    os.makedirs('models', exist_ok=True)
    
    # Save the math vectors and labels
    joblib.dump(X, 'data/X_features.joblib')
    joblib.dump(y, 'data/y_labels.joblib')
    
    # CRITICAL: We must save the vectorizer tool itself so our API can use it later
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.joblib')
    
    print("Stage 2 Complete! Features and Vectorizer saved.")