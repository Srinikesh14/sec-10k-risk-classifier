from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from src.preprocess import clean_text

app = FastAPI(title="SEC 10-K Risk Classifier API")

# Load our saved tools from Stage 2 and 4
vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
model = joblib.load('models/best_model.joblib')

# Define the exact input format requested in the assignment
class DocumentInput(BaseModel):
    text: str

@app.post("/predict")
def predict_risk(doc: DocumentInput):
    # 1. Clean the incoming raw text exactly like we did in training
    cleaned = clean_text(doc.text)
    
    # 2. Convert text to numbers using our saved TF-IDF vectorizer
    features = vectorizer.transform([cleaned])
    
    # 3. Predict the label and confidence (probability)
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    # Confidence is the probability of the chosen class
    confidence = float(max(probabilities))
    
    # Map the numeric prediction back to a readable label
    label = "High Risk" if prediction == 1 else "Normal Risk"
    
    # Output the exact JSON structure required
    return {
        "label": label,
        "confidence": round(confidence, 4)
    }