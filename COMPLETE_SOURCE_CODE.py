"""
COMPLETE SOURCE CODE FOR HACK4HEALTH
========================================
"""



# ========================================
# FILE: config.py
# ========================================

import os
BASE_DIR = r"c:\Users\milin\Downloads\Hack Health"
DATA_FILE = os.path.join(BASE_DIR, "Primary_Emotions.xlsx - Sheet1.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "emotion_classifier", "outputs")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
SEED = 42


# ========================================
# FILE: predict.py
# ========================================

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
        try:
            probs = model.predict_proba(vec)[0]
            max_prob = max(probs)
            confidence = f"{max_prob:.2%}"
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


# ========================================
# FILE: scripts/1_data_loading.py
# ========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config
import os
def load_and_inspect_data():
    print(">>> PHASE 1: Data Loading & Understanding")
    print(f"Loading data from: {config.DATA_FILE}")
    try:
        df = pd.read_csv(config.DATA_FILE)
    except Exception as e:
        print(f"Error loading CSV with default encoding: {e}")
        print("Retrying with 'utf-8-sig'...")
        df = pd.read_csv(config.DATA_FILE, encoding='utf-8-sig')
    df.columns = [col.strip().replace('\n', '') for col in df.columns]
    print(f"Columns found: {df.columns.tolist()}")
    text_col = 'Poem'
    label_col = [c for c in df.columns if 'Primary' in c][0]
    source_col = 'Source'
    print(f"Target Text Column: '{text_col}'")
    print(f"Target Label Column: '{label_col}'")
    df = df.rename(columns={label_col: 'Label'})
    print(f"\nDataset Shape: {df.shape}")
    print("\nSample Data:")
    print(df.head())
    print("\n--- Class Distribution ---")
    print(df['Label'].value_counts())
    if source_col in df.columns:
        print("\n--- Language/Source Distribution ---")
        print(df[source_col].value_counts())
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    df = df.dropna(subset=[text_col, 'Label'])
    print(f"\nShape after dropping nulls: {df.shape}")
    plt.figure(figsize=(10, 6))
    sns.countplot(y=df['Label'], order=df['Label'].value_counts().index, palette='viridis')
    plt.title('Emotion Class Distribution')
    plt.xlabel('Count')
    plt.ylabel('Emotion')
    plt.tight_layout()
    plot_path = os.path.join(config.IMAGES_DIR, "class_distribution.png")
    plt.savefig(plot_path)
    print(f"\nSaved distribution plot to: {plot_path}")
    print("\n--- Text Samples (Unicode Check) ---")
    for i, row in df.head(5).iterrows():
        print(f"Label: {row['Label']} | Source: {row.get(source_col, 'N/A')} | Text snippet: {row[text_col][:50]}...")
    return df
if __name__ == "__main__":
    load_and_inspect_data()


# ========================================
# FILE: scripts/2_preprocessing.py
# ========================================

import pandas as pd
import re
import config
from phase1_data_loading import load_and_inspect_data
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
def preprocess_dataset():
    print(">>> PHASE 2: Language-Safe Preprocessing")
    df = load_and_inspect_data()
    print("\nApplying text cleaning...")
    df['Clean_Text'] = df['Poem'].apply(clean_text)
    print("\n--- Data Cleaning Comparison ---")
    sample = df.sample(1).iloc[0]
    print(f"Original: {sample['Poem']}")
    print(f"Cleaned : {sample['Clean_Text']}")
    print(f"\nShape after preprocessing: {df.shape}")
    processed_path = os.path.join(config.OUTPUT_DIR, "processed_data.csv")
    df.to_csv(processed_path, index=False, encoding='utf-8-sig')
    print(f"Saved processed data to: {processed_path}")
    return df
import os
if __name__ == "__main__":
    preprocess_dataset()


# ========================================
# FILE: scripts/3_baseline_model.py
# ========================================

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


# ========================================
# FILE: scripts/4_semantic_model.py
# ========================================

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
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.title('Semantic Model (SBERT Multilingual) Confusion Matrix')
    plt.savefig(os.path.join(config.IMAGES_DIR, "confusion_matrix_semantic_multi.png"))
    print("Saved Confusion Matrix.")
if __name__ == "__main__":
    train_semantic_model_local()


# ========================================
# FILE: scripts/4b_mlp_classifier.py
# ========================================

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
    model_path = os.path.join(config.OUTPUT_DIR, "sbert_mlp_model.pkl")
    joblib.dump(clf, model_path)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.title('SBERT + MLP Confusion Matrix')
    plt.savefig(os.path.join(config.IMAGES_DIR, "confusion_matrix_sbert_mlp.png"))
if __name__ == "__main__":
    train_sbert_mlp()


# ========================================
# FILE: scripts/5_transformer_model.py
# ========================================

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import joblib
import config
import os
class CustomTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.args.device)
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {'accuracy': acc, 'f1_weighted': f1}
def train_gpu_model_local():
    print(">>> PHASE 5: GPU-Based Transformer Model (Local XLM-R)")
    if not torch.cuda.is_available():
        print("WARNING: GPU not available. Using CPU (Will be very slow).")
        device = 'cpu'
    else:
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    data_path = os.path.join(config.OUTPUT_DIR, "processed_data.csv")
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['Clean_Text', 'Label'])
    X = df['Clean_Text'].astype(str).tolist()
    y_raw = df['Label'].tolist()
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = le.classes_
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.SEED, stratify=y
    )
    local_model_path = os.path.join(config.BASE_DIR, "emotion_classifier", "models", "xlmr")
    print(f"Loading Local Model from: {local_model_path}")
    try:
        tokenizer = XLMRobertaTokenizer.from_pretrained(local_model_path)
        model = XLMRobertaForSequenceClassification.from_pretrained(local_model_path, num_labels=len(classes), ignore_mismatched_sizes=True)
    except Exception as e:
        print(f"Failed to load local model: {e}")
        return
    print("Tokenizing data...")
    train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=128)
    train_dataset = EmotionDataset(train_encodings, y_train)
    test_dataset = EmotionDataset(test_encodings, y_test)
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    class_weights = total_samples / (len(classes) * class_counts)
    training_args = TrainingArguments(
        output_dir=os.path.join(config.OUTPUT_DIR, 'results_xlm_roberta'),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=os.path.join(config.OUTPUT_DIR, 'logs'),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True if device == 'cuda' else False,
        no_cuda=False if device == 'cuda' else True,
        report_to="none"
    )
    trainer = CustomTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    print("Starting Training...")
    trainer.train()
    print("Evaluating...")
    trainer.evaluate()
    model_save_path = os.path.join(config.OUTPUT_DIR, "xlmr_model")
    trainer.save_model(model_save_path)
    joblib.dump(le, os.path.join(model_save_path, "label_encoder.pkl"))
    print("Saved XLM-R model.")
    print("Generating predictions on Test Set...")
    predictions = trainer.predict(test_dataset)
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()
    np.save(os.path.join(config.OUTPUT_DIR, "xlmr_probs.npy"), probs)
if __name__ == "__main__":
    train_gpu_model_local()


# ========================================
# FILE: scripts/6_ensemble_model.py
# ========================================

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
    X_train, X_test, y_train, y_test = load_data_and_split()
    print(f"Test Set Size: {len(X_test)}")
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
    xlmr_probs_path = os.path.join(config.OUTPUT_DIR, "xlmr_probs.npy")
    if os.path.exists(xlmr_probs_path):
        print("Loading XLM-R Probabilities...")
        probs_xlmr = np.load(xlmr_probs_path)
        if probs_xlmr.shape[0] != len(y_test):
             print("WARNING: XLM-R shape mismatch. Skipping.")
             probs_xlmr = None
    else:
        print("XLM-R probabilities not found. Skipping.")
    print("Calculating Ensemble Predictions...")
    valid_probs = [probs_baseline]
    if probs_semantic is not None: valid_probs.append(probs_semantic)
    if probs_xlmr is not None: valid_probs.append(probs_xlmr)
    ensemble_probs = sum(valid_probs) / len(valid_probs)
    y_pred_ensemble = classes[np.argmax(ensemble_probs, axis=1)]
    acc = accuracy_score(y_test, y_pred_ensemble)
    f1 = f1_score(y_test, y_pred_ensemble, average='weighted', labels=classes)
    f1 = f1_score(y_test, y_pred_ensemble, average='weighted')
    print(f"\nEnsemble Accuracy: {acc:.4f}")
    print(f"Ensemble Weighted F1: {f1:.4f}")
    print("\n--- Ensemble Classification Report ---")
    print(classification_report(y_test, y_pred_ensemble))
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


# ========================================
# FILE: scripts/7_8_error_analysis.py
# ========================================

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
    data_path = os.path.join(config.OUTPUT_DIR, "processed_data.csv")
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['Clean_Text', 'Label'])
    baseline_model_path = os.path.join(config.OUTPUT_DIR, "baseline_model.pkl")
    baseline_tfidf_path = os.path.join(config.OUTPUT_DIR, "baseline_tfidf.pkl")
    if not os.path.exists(baseline_model_path):
        print("Baseline model not found! Run Phase 3.")
        return
    model = joblib.load(baseline_model_path)
    tfidf = joblib.load(baseline_tfidf_path)
    print("Generating predictions using Baseline Model for Analysis...")
    X = df['Clean_Text']
    y = df['Label']
    X_vec = tfidf.transform(X)
    y_pred = model.predict(X_vec)
    cm = confusion_matrix(y, y_pred, labels=model.classes_)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Confusion Matrix (Baseline - Full Dataset)')
    plt.tight_layout()
    plt.savefig(os.path.join(config.IMAGES_DIR, "confusion_matrix_full_baseline.png"))
    print("Saved Baseline Confusion Matrix.")
    embeddings_path = os.path.join(config.OUTPUT_DIR, "embeddings.npy")
    if os.path.exists(embeddings_path):
        print("Loading SBERT embeddings for t-SNE...")
        X_emb = np.load(embeddings_path)
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
    print("\n--- Error Analysis ---")
    df['Predicted'] = y_pred
    misclassified = df[df['Label'] != df['Predicted']]
    print(f"Total Misclassified: {len(misclassified)} / {len(df)}")
    print("\nCommon Misclassifications:")
    print(misclassified.groupby(['Label', 'Predicted']).size().sort_values(ascending=False).head(10))
    print("\nSample Misclassified Examples:")
    if not misclassified.empty:
        sample_mis = misclassified.sample(min(5, len(misclassified)))
        for idx, row in sample_mis.iterrows():
            print(f"-"*80)
            print(f"Text: {row['Poem'][:200]}...")
            print(f"True: {row['Label']} | Pred: {row['Predicted']}")
    print("\nAnalysis Complete.")
if __name__ == "__main__":
    analyze_models()


# ========================================
# FILE: scripts/9_performance_report.py
# ========================================

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
        clf = joblib.load(os.path.join(config.OUTPUT_DIR, model_pkl_name))
        sbert_path = os.path.join(config.BASE_DIR, "emotion_classifier", "models", folder_name)
        sbert = SentenceTransformer(sbert_path)
        embeddings = sbert.encode(X_test.tolist(), show_progress_bar=False)
        y_pred = clf.predict(embeddings)
        return f1_score(y_test, y_pred, average='weighted')
    except Exception as e:
        print(f"Error eval {folder_name}: {e}")
        return None
def evaluate_sbert_mlp(X_test, y_test):
    try:
        clf = joblib.load(os.path.join(config.OUTPUT_DIR, "sbert_mlp_model.pkl"))
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
