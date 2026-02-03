import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
import joblib
import config
import os

def train_baseline_model():
    print(">>> PHASE 3: Baseline Model (TF-IDF + SVM)")
    
    data_path = os.path.join(config.OUTPUT_DIR, "processed_data.csv")
    if not os.path.exists(data_path):
        print("Processed data not found. Run Phase 2 first.")
        return

    df = pd.read_csv(data_path)
    df = df.dropna(subset=['Clean_Text', 'Label'])
    
    X = df['Clean_Text']
    y = df['Label']
    
    print(f"Data loaded. Samples: {len(df)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.SEED, stratify=y
    )
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    
    print("\nTraining TF-IDF (Char 3-5 ngrams)...")
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=3)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)
    
    print(f"Vocabulary size: {len(tfidf.vocabulary_)}")
    
    print("Training Linear SVM (Balanced)...")
    svm = LinearSVC(class_weight='balanced', random_state=config.SEED, dual='auto')
    model = CalibratedClassifierCV(svm) 
    model.fit(X_train_vec, y_train)
    
    print("\nEvaluating...")
    y_pred = model.predict(X_test_vec)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}")
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    print("\nGenerating Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Baseline (TF-IDF + SVM) Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    plot_path = os.path.join(config.IMAGES_DIR, "confusion_matrix_baseline.png")
    plt.savefig(plot_path)
    print(f"Saved confusion matrix to: {plot_path}")
    
    model_path = os.path.join(config.OUTPUT_DIR, "baseline_model.pkl")
    vectorizer_path = os.path.join(config.OUTPUT_DIR, "baseline_tfidf.pkl")
    
    joblib.dump(model, model_path)
    joblib.dump(tfidf, vectorizer_path)
    print("Saved model and vectorizer.")

if __name__ == "__main__":
    train_baseline_model()
