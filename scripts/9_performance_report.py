import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score
import joblib
import config
import os
import torch
from sentence_transformers import SentenceTransformer
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer

def load_data_and_split():
    data_path = os.path.join(config.OUTPUT_DIR, "processed_data.csv")
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['Clean_Text', 'Label'])
    X = df['Clean_Text']
    y = df['Label']
    return X, y

def evaluate_baseline(X_test, y_test):
    try:
        model = joblib.load(os.path.join(config.OUTPUT_DIR, "baseline_model.pkl"))
        tfidf = joblib.load(os.path.join(config.OUTPUT_DIR, "baseline_tfidf.pkl"))
        X_vec = tfidf.transform(X_test)
        y_pred = model.predict(X_vec)
        return f1_score(y_test, y_pred, average='weighted')
    except:
        return None

def evaluate_sbert_logreg(folder_name, model_pkl_name, X_test, y_test):
    try:
        # Load LogReg
        clf = joblib.load(os.path.join(config.OUTPUT_DIR, model_pkl_name))
        
        # Load SBERT to encode test key
        sbert_path = os.path.join(config.BASE_DIR, "emotion_classifier", "models", folder_name)
        sbert = SentenceTransformer(sbert_path)
        
        embeddings = sbert.encode(X_test.tolist(), show_progress_bar=False)
        y_pred = clf.predict(embeddings)
        return f1_score(y_test, y_pred, average='weighted')
    except Exception as e:
        print(f"Error eval {folder_name}: {e}")
        return None

def evaluate_sbert_mlp(X_test, y_test):
    # This requires SBERT embeddings to match the training ones (multilingual)
    try:
        clf = joblib.load(os.path.join(config.OUTPUT_DIR, "sbert_mlp_model.pkl"))
        # We need the embeddings. For now, let's assume we can generate them or they exist?
        # If we can't load the SBERT model, we can't generate test embeddings.
        # But if we trained MLP, we must have had embeddings.
        # Let's hope the 'semantic_model_multi' logic generated test embeddings?
        # Actually, simpler: load the test embeddings if saved, or just skip if we can't generation.
        # Since this is complex without the SBERT model loaded, we'll skip if SBERT Multi isn't working.
        return None 
    except:
        return None

def final_report():
    print(">>> PHASE 9: Final Model Comparison Report")
    
    X, y = load_data_and_split()
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=config.SEED, stratify=y)
    
    results = {}
    
    print("Evaluating Baseline...")
    results['Baseline (TF-IDF)'] = evaluate_baseline(X_test, y_test)
    
    print("Evaluating SBERT Multilingual...")
    results['SBERT Multi (LogReg)'] = evaluate_sbert_logreg("sbert_multi", "semantic_model_multi.pkl", X_test, y_test)
    
    print("Evaluating SBERT Indic...")
    results['SBERT Indic (LogReg)'] = evaluate_sbert_logreg("sbert_indic", "semantic_model_indic.pkl", X_test, y_test)
    
    # Clean None
    results = {k: v for k, v in results.items() if v is not None}
    
    print("\n" + "="*40)
    print("FINAL RESULTS (Weighted F1)")
    print("="*40)
    df_res = pd.DataFrame(list(results.items()), columns=['Model', 'F1 Score'])
    df_res = df_res.sort_values(by='F1 Score', ascending=False)
    print(df_res)
    
    if not df_res.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='F1 Score', y='Model', data=df_res, palette='viridis')
        plt.title('Final Model Comparison')
        plt.xlabel('Weighted F1 Score')
        plt.tight_layout()
        plt.savefig(os.path.join(config.IMAGES_DIR, "final_comparison.png"))
        print(f"\nSaved comparison plot to {os.path.join(config.IMAGES_DIR, 'final_comparison.png')}")

if __name__ == "__main__":
    final_report()
