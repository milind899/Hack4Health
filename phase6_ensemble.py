import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sentence_transformers import SentenceTransformer
import joblib
import config
import os

def load_data_and_split():
    data_path = os.path.join(config.OUTPUT_DIR, "processed_data.csv")
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['Clean_Text', 'Label'])
    X = df['Clean_Text']
    y = df['Label']
    return train_test_split(X, y, test_size=0.2, random_state=config.SEED, stratify=y)

def run_ensemble():
    print(">>> PHASE 6: Model Ensembling (Robust)")
    
    # 1. Load Data
    X_train, X_test, y_train, y_test = load_data_and_split()
    print(f"Test Set Size: {len(X_test)}")
    
    # 2. Get Baseline Probabilities
    print("Loading Baseline Model...")
    try:
        baseline_model = joblib.load(os.path.join(config.OUTPUT_DIR, "baseline_model.pkl"))
        baseline_tfidf = joblib.load(os.path.join(config.OUTPUT_DIR, "baseline_tfidf.pkl"))
    except Exception as e:
        print(f"Error loading Baseline: {e}")
        return

    X_test_vec = baseline_tfidf.transform(X_test)
    probs_baseline = baseline_model.predict_proba(X_test_vec)
    classes = baseline_model.classes_
    print(f"Baseline probabilities shape: {probs_baseline.shape}")
    
    probs_semantic = None
    probs_xlmr = None

    # 3. Get Semantic Probabilities
    semantic_path = os.path.join(config.OUTPUT_DIR, "semantic_model_multi.pkl")
    if os.path.exists(semantic_path):
        try:
            print("Loading Semantic Model...")
            semantic_model = joblib.load(semantic_path)
            print("Encoding Test Set for Semantic Model...")
            local_sbert_path = os.path.join(config.BASE_DIR, "emotion_classifier", "models", "sbert_multi")
            sbert = SentenceTransformer(local_sbert_path)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            X_test_emb = sbert.encode(X_test.tolist(), show_progress_bar=True, device=device)
            probs_semantic = semantic_model.predict_proba(X_test_emb)
            print(f"Semantic probabilities shape: {probs_semantic.shape}")
            
            if not np.array_equal(classes, semantic_model.classes_):
                print("WARNING: Class mismatch! Skipping Semantic.")
                probs_semantic = None
        except Exception as e:
            print(f"Error Semantic: {e}")
            probs_semantic = None
    else:
        print("Semantic model not found (Phase 4 incomplete). Skipping.")

    # 4. Get XLM-R Probabilities
    xlmr_probs_path = os.path.join(config.OUTPUT_DIR, "xlmr_probs.npy")
    if os.path.exists(xlmr_probs_path):
        print("Loading XLM-R Probabilities...")
        probs_xlmr = np.load(xlmr_probs_path)
        if probs_xlmr.shape[0] != len(y_test):
             print("WARNING: XLM-R shape mismatch. Skipping.")
             probs_xlmr = None
    else:
        print("XLM-R probabilities not found. Skipping.")
        
    # 5. Soft Voting
    print("Calculating Ensemble Predictions...")
    
    valid_probs = [probs_baseline]
    if probs_semantic is not None: valid_probs.append(probs_semantic)
    if probs_xlmr is not None: valid_probs.append(probs_xlmr)
    
    ensemble_probs = sum(valid_probs) / len(valid_probs)
    y_pred_ensemble = classes[np.argmax(ensemble_probs, axis=1)]
    
    # 6. Evaluation
    acc = accuracy_score(y_test, y_pred_ensemble)
    f1 = f1_score(y_test, y_pred_ensemble, average='weighted', labels=classes)
    f1 = f1_score(y_test, y_pred_ensemble, average='weighted')

    print(f"\nEnsemble Accuracy: {acc:.4f}")
    print(f"Ensemble Weighted F1: {f1:.4f}")
    
    print("\n--- Ensemble Classification Report ---")
    print(classification_report(y_test, y_pred_ensemble))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_ensemble, labels=classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Ensemble Model Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(config.IMAGES_DIR, "confusion_matrix_ensemble.png"))
    print("Saved Ensemble Confusion Matrix.")

if __name__ == "__main__":
    run_ensemble()
