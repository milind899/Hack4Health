import joblib
import pandas as pd
import config
import os
import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_emotion():
    print(">>> Interactive Emotion Prediction (Baseline Model)")
    
    model_path = os.path.join(config.OUTPUT_DIR, "baseline_model.pkl")
    vectorizer_path = os.path.join(config.OUTPUT_DIR, "baseline_tfidf.pkl")
    
    if not os.path.exists(model_path):
        print("Model not found! Run Phase 3 first.")
        return

    print("Loading model...")
    model = joblib.load(model_path)
    tfidf = joblib.load(vectorizer_path)
    print("Model loaded successfully.\n")
    
    print("-" * 50)
    print("Type a poem or verse (Tamil/Bengali/English).")
    print("Type 'exit' to quit.")
    print("-" * 50)
    
    while True:
        text = input("\nEnter text: ")
        if text.lower() == 'exit':
            break
        
        cleaned = clean_text(text)
        if not cleaned:
            print("Empty input!")
            continue
            
        vec = tfidf.transform([cleaned])
        pred = model.predict(vec)[0]
        
        # Get probability/confidence
        try:
            probs = model.predict_proba(vec)[0]
            max_prob = max(probs)
            confidence = f"{max_prob:.2%}"
            
            # Show top 3
            class_probs = list(zip(model.classes_, probs))
            class_probs.sort(key=lambda x: x[1], reverse=True)
            top3 = class_probs[:3]
            top3_str = ", ".join([f"{c} ({p:.1%})" for c, p in top3])
            
        except AttributeError:
            confidence = "N/A (Model doesn't support proba)"
            top3_str = "N/A"
        
        print(f"\nPredicted Emotion: {pred.upper()}")
        print(f"Confidence: {confidence}")
        print(f"Top 3 candidates: {top3_str}")

if __name__ == "__main__":
    predict_emotion()
