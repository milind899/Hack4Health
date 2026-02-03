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
