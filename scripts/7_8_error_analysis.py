import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import config
import os
import random

def analyze_models():
    print(">>> PHASE 7 & 8: Visualization & Error Analysis")
    
    # 1. Load Data
    data_path = os.path.join(config.OUTPUT_DIR, "processed_data.csv")
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['Clean_Text', 'Label'])
    
    # Load Models
    baseline_model_path = os.path.join(config.OUTPUT_DIR, "baseline_model.pkl")
    baseline_tfidf_path = os.path.join(config.OUTPUT_DIR, "baseline_tfidf.pkl")
    
    if not os.path.exists(baseline_model_path):
        print("Baseline model not found! Run Phase 3.")
        return

    model = joblib.load(baseline_model_path)
    tfidf = joblib.load(baseline_tfidf_path)
    
    # Generate Predictions (Baseline)
    print("Generating predictions using Baseline Model for Analysis...")
    X = df['Clean_Text']
    y = df['Label']
    
    X_vec = tfidf.transform(X)
    y_pred = model.predict(X_vec)
    
    # 2. Confusion Matrix (Full Dataset)
    cm = confusion_matrix(y, y_pred, labels=model.classes_)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Confusion Matrix (Baseline - Full Dataset)')
    plt.tight_layout()
    plt.savefig(os.path.join(config.IMAGES_DIR, "confusion_matrix_full_baseline.png"))
    print("Saved Baseline Confusion Matrix.")
    
    # 3. Embedding Visualization (t-SNE)
    # Use SBERT embeddings if available, else TF-IDF (PCA reduced)
    embeddings_path = os.path.join(config.OUTPUT_DIR, "embeddings.npy")
    
    if os.path.exists(embeddings_path):
        print("Loading SBERT embeddings for t-SNE...")
        X_emb = np.load(embeddings_path)
        # Use a subset if too large
        if len(X_emb) > 2000:
            indices = np.random.choice(len(X_emb), 2000, replace=False)
            X_vis = X_emb[indices]
            y_vis = y.iloc[indices]
        else:
            X_vis = X_emb
            y_vis = y
            
        print("Running t-SNE...")
        tsne = TSNE(n_components=2, random_state=config.SEED, init='pca', learning_rate='auto')
        X_tsne = tsne.fit_transform(X_vis)
        
        plt.figure(figsize=(12, 10))
        sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=y_vis, palette='tab10', s=10, alpha=0.7)
        plt.title('t-SNE Visualization of SBERT Embeddings')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(config.IMAGES_DIR, "tsne_embeddings.png"))
        print("Saved t-SNE plot.")
        
    else:
        print("SBERT embeddings not found. Skipping t-SNE.")

    # 4. Error Analysis
    print("\n--- Error Analysis ---")
    df['Predicted'] = y_pred
    misclassified = df[df['Label'] != df['Predicted']]
    
    print(f"Total Misclassified: {len(misclassified)} / {len(df)}")
    
    print("\nCommon Misclassifications:")
    print(misclassified.groupby(['Label', 'Predicted']).size().sort_values(ascending=False).head(10))
    
    print("\nSample Misclassified Examples:")
    # Pick 5 random
    if not misclassified.empty:
        sample_mis = misclassified.sample(min(5, len(misclassified)))
        for idx, row in sample_mis.iterrows():
            print(f"-"*80)
            print(f"Text: {row['Poem'][:200]}...")
            print(f"True: {row['Label']} | Pred: {row['Predicted']}")
            
    print("\nAnalysis Complete.")

if __name__ == "__main__":
    analyze_models()
