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

def train_semantic_model_local():
    print(">>> PHASE 4: Semantic Representation (Local SBERT Multilingual)")
    
    data_path = os.path.join(config.OUTPUT_DIR, "processed_data.csv")
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['Clean_Text', 'Label'])
    
    X_text = df['Clean_Text'].astype(str).tolist()
    y = df['Label']
    
    # Load Local Model
    local_model_path = os.path.join(config.BASE_DIR, "emotion_classifier", "models", "sbert_multi")
    print(f"Loading SBERT model from local path: {local_model_path}")
    
    try:
        sbert_model = SentenceTransformer(local_model_path)
    except Exception as e:
        print(f"Failed to load local model: {e}")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    sbert_model.to(device)
    
    print("Generating embeddings...")
    X_embeddings = sbert_model.encode(X_text, show_progress_bar=True, device=device)
    
    # Save embeddings
    embeddings_path = os.path.join(config.OUTPUT_DIR, "embeddings_multi.npy")
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
    
    acc = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}")
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    model_path = os.path.join(config.OUTPUT_DIR, "semantic_model_multi.pkl")
    joblib.dump(clf, model_path)
    print("Saved semantic model.")
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.title('Semantic Model (SBERT Multilingual) Confusion Matrix')
    plt.savefig(os.path.join(config.IMAGES_DIR, "confusion_matrix_semantic_multi.png"))
    print("Saved Confusion Matrix.")

if __name__ == "__main__":
    train_semantic_model_local()
