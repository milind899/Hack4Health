import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
import config
import os

def train_sbert_mlp():
    print(">>> PHASE 4b: Neural Network (MLP) on SBERT Embeddings")
    
    embeddings_path = os.path.join(config.OUTPUT_DIR, "embeddings_multi.npy")
    labels_path = os.path.join(config.OUTPUT_DIR, "labels.pkl")
    
    if not os.path.exists(embeddings_path):
        print("Embeddings not found! Run Phase 4 Local first.")
        return

    X = np.load(embeddings_path)
    y = joblib.load(labels_path)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.SEED, stratify=y
    )
    
    print("Training MLP Classifier (Hidden Layers: 256, 128)...")
    # MLP with Early Stopping
    clf = MLPClassifier(hidden_layer_sizes=(256, 128),
                        activation='relu',
                        solver='adam',
                        alpha=0.0001,
                        batch_size=32,
                        learning_rate='adaptive',
                        early_stopping=True,
                        validation_fraction=0.1,
                        max_iter=500,
                        random_state=config.SEED,
                        verbose=True)
                        
    clf.fit(X_train, y_train)
    
    print("\nEvaluating...")
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"accuracy: {acc:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}")
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    # Save
    model_path = os.path.join(config.OUTPUT_DIR, "sbert_mlp_model.pkl")
    joblib.dump(clf, model_path)
    
    # Plot CM
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.title('SBERT + MLP Confusion Matrix')
    plt.savefig(os.path.join(config.IMAGES_DIR, "confusion_matrix_sbert_mlp.png"))

if __name__ == "__main__":
    train_sbert_mlp()
