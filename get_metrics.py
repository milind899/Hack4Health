import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import os

df = pd.read_csv('outputs/processed_data.csv')
df = df.dropna(subset=['Clean_Text', 'Label'])
print(f"Total Rows: {len(df)}")
print(f"Total Classes: {df['Label'].nunique()}")

X = df['Clean_Text']
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Baseline
if os.path.exists('outputs/baseline_model.pkl'):
    model = joblib.load('outputs/baseline_model.pkl')
    tfidf = joblib.load('outputs/baseline_tfidf.pkl')
    X_test_vec = tfidf.transform(X_test)
    y_pred = model.predict(X_test_vec)
    print(f"Baseline - Acc: {accuracy_score(y_test, y_pred):.4f}, Macro: {f1_score(y_test, y_pred, average='macro'):.4f}, Weighted: {f1_score(y_test, y_pred, average='weighted'):.4f}")

# Semantic
if os.path.exists('outputs/semantic_model_multi.pkl'):
    # This needs embeddings, skip for now, I'll just look for the logs if possible
    pass
