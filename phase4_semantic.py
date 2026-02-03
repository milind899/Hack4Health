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

def train_semantic_model():
    print(">>> PHASE 4: Semantic Representation Upgrade (SBERT)")
    
    data_path = os.path.join(config.OUTPUT_DIR, "processed_data.csv")
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['Clean_Text', 'Label'])
    
    X_text = df['Clean_Text'].astype(str).tolist()
    y = df['Label']
    
    print("Loading SBERT model (paraphrase-multilingual-MiniLM-L12-v2)...")
    sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    print("Generating embeddings (this may take a while)...")
    X_embeddings = sbert_model.encode(X_text, show_progress_bar=True)
    
    print(f"Embeddings shape: {X_embeddings.shape}")
    
    # Save embeddings for Ensemble later
    embeddings_path = os.path.join(config.OUTPUT_DIR, "embeddings.npy")
    np.save(embeddings_path, X_embeddings)
    joblib.dump(y, os.path.join(config.OUTPUT_DIR, "labels.pkl"))
    print("Saved embeddings and labels.")

    X_train, X_test, y_train, y_test = train_test_split(
        X_embeddings, y, test_size=0.2, random_state=config.SEED, stratify=y
    )
    
    print("Training Logistic Regression on Embeddings...")
    clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=config.SEED)
    clf.fit(X_train, y_train)
    
    print("\nEvaluating...")
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}")
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    print("\nGenerating Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.title('Semantic Model (SBERT + LogReg) Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    plot_path = os.path.join(config.IMAGES_DIR, "confusion_matrix_semantic.png")
    plt.savefig(plot_path)
    print(f"Saved confusion matrix to: {plot_path}")
    
    model_path = os.path.join(config.OUTPUT_DIR, "semantic_model.pkl")
    joblib.dump(clf, model_path)
    print("Saved semantic model.")

if __name__ == "__main__":
    train_semantic_model()
