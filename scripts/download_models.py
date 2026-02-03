import os
import shutil
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, AutoConfig

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

def download_sbert(model_name, folder_name):
    print(f"\nDownloading {model_name}...")
    target_path = os.path.join(MODELS_DIR, folder_name)
    
    # Clean if exists to ensure fresh download
    # if os.path.exists(target_path):
    #     shutil.rmtree(target_path)
    
    try:
        model = SentenceTransformer(model_name)
        model.save(target_path)
        print(f"Successfully saved to {target_path}")
    except Exception as e:
        print(f"Failed to download {model_name}: {e}")

def download_transformer(model_name, folder_name):
    print(f"\nDownloading {model_name}...")
    target_path = os.path.join(MODELS_DIR, folder_name)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        tokenizer.save_pretrained(target_path)
        model.save_pretrained(target_path)
        print(f"Successfully saved to {target_path}")
    except Exception as e:
        print(f"Failed to download {model_name}: {e}")

if __name__ == "__main__":
    print("Starting automated download...")
    
    # 1. Multilingual SBERT
    download_sbert("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "sbert_multi")
    
    # 2. Indic SBERT
    download_sbert("l3cube-pune/indic-sentence-bert-nli", "sbert_indic")
    
    # 3. XLM-RoBERTa
    download_transformer("xlm-roberta-base", "xlmr")
    
    print("\nDownload process finished.")
