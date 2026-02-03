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
