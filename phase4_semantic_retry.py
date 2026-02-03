import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sentence_transformers import SentenceTransformer
import joblib
import config
import os
import torch

def train_semantic_model():
    print(">>> PHASE 4: Semantic Representation Upgrade (SBERT) - Attempt 2")
    
    data_path = os.path.join(config.OUTPUT_DIR, "processed_data.csv")
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['Clean_Text', 'Label'])
    
    X_text = df['Clean_Text'].astype(str).tolist()
    y = df['Label']
    
    model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
    print(f"Loading SBERT model ({model_name})...")
    
    try:
        sbert_model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        print("Trying fallback: 'all-MiniLM-L6-v2' (English-focused but works OK usually)")
        try:
             sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e2:
             print(f"Critical Error: Could not download any model. {e2}")
             return

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    sbert_model.to(device)
    
    print("Generating embeddings...")
    X_embeddings = sbert_model.encode(X_text, show_progress_bar=True, device=device)
    
    # Save embeddings
    embeddings_path = os.path.join(config.OUTPUT_DIR, "embeddings.npy")
    np.save(embeddings_path, X_embeddings)
    joblib.dump(y, os.path.join(config.OUTPUT_DIR, "labels.pkl"))
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_embeddings, y, test_size=0.2, random_state=config.SEED, stratify=y
    )
    
    print("Training Logistic Regression on Embeddings...")
    clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=config.SEED, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    print("\nEvaluating...")
    y_pred = clf.predict(X_test)
    
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    print(f"Weighted F1: {f1_weighted:.4f}")
    
    model_path = os.path.join(config.OUTPUT_DIR, "semantic_model.pkl")
    joblib.dump(clf, model_path)
    
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.savefig(os.path.join(config.IMAGES_DIR, "confusion_matrix_semantic.png"))
    print("Complete.")

if __name__ == "__main__":
    train_semantic_model()
