# Model Download Instructions (Final List for Comparison)

For a robust comparison, we will use these 3 downloaded models + the TF-IDF Baseline.

## 1. Benchmark Semantic Model (Multilingual SBERT)
**Target Folder:** `emotion_classifier/models/sbert_multi`
**Model:** `paraphrase-multilingual-MiniLM-L12-v2`
**URL:** [https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/tree/main](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/tree/main)

**Required Files:** `config.json`, `pytorch_model.bin`, `tokenizer.json`, `vocab.txt`, `special_tokens_map.json`, `tokenizer_config.json`, `modules.json`, `sentence_bert_config.json`

## 2. Optimized Indic Model (L3Cube-IndicSBERT)
**Target Folder:** `emotion_classifier/models/sbert_indic`
**Model:** `l3cube-pune/indic-sentence-bert-nli`
**URL:** [https://huggingface.co/l3cube-pune/indic-sentence-bert-nli/tree/main](https://huggingface.co/l3cube-pune/indic-sentence-bert-nli/tree/main)

**Required Files:** Same as above.

## 3. High-Performance GPU Model (XLM-RoBERTa)
**Target Folder:** `emotion_classifier/models/xlmr`
**Model:** `xlm-roberta-base`
**URL:** [https://huggingface.co/xlm-roberta-base/tree/main](https://huggingface.co/xlm-roberta-base/tree/main)

**Required Files:** `config.json`, `pytorch_model.bin`, `sentencepiece.bpe.model`, `tokenizer.json`

---
**Comparison Plan:**
1.  **Baseline**: TF-IDF (Fast, interpretable)
2.  **SBERT Multi**: Global standard for sentence embeddings.
3.  **SBERT Indic**: Performance on Tamil/Bengali specifically.
4.  **XLM-R**: Impact of fine-tuning a large transformer.

Please download files for the 3 folders above.
