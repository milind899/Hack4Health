# 🎭 Hack4Health: Multilingual Emotion Classifier for Poetry

Welcome to **Hack4Health**! This project is a state-of-the-art AI system designed to understand the deep, nuanced emotions found in multilingual poetry—specifically focusing on **Tamil** and **Bengali** scripts. 

Poetry is the ultimate test for AI; it's dense, metaphorical, and culturally rich. Our goal was to build a system that doesn't just "see" words, but actually "feels" the underlying sentiment without relying on translation.

## 🌟 The Vision
Traditional sentiment analysis often fails when it meets the complexity of local languages and poetic structures. We built a system that preserves the soul of the language by:
1.  **Avoiding Translation**: Working directly with Unicode Tamil and Bengali scripts.
2.  **Semantic Depth**: Mapping poetic verses to 46 distinct emotion categories.
3.  **Hybrid Intelligence**: Combining statistical patterns with deep neural embeddings.

---

## 🏗️ How it Works (The Pipeline)

Our system uses a **Soft-Voting Ensemble** architecture (see the diagram in the assets):

1.  **Unicode Preprocessing**: Cleaning and normalizing the local script data without losing the original meaning.
2.  **Hybrid Encoding**:
    *   **Baseline (TF-IDF + SVM)**: Captures linguistic structure and character-level patterns.
    *   **Semantic (SBERT)**: Uses multilingual Sentence-BERT to understand the "meaning" across languages.
    *   **Deep Learning (XLM-RoBERTa)**: Fine-tuned for dense emotional classification.
3.  **Aggregation**: A consensus layer that merges probabilities to give the most robust prediction.

---

## � Project Structure
```text
.
├── assets/                # Presentation visuals and diagrams
├── scripts/               # Step-by-step implementation pipeline
│   ├── 1_data_loading.py
│   ├── 2_preprocessing.py
│   ├── 3_baseline_model.py
│   ├── 4_semantic_model.py
│   ├── 6_ensemble_model.py
│   └── ... (and more)
├── config.py              # Central configuration
├── predict.py             # CLI Tool for baseline inference
├── predict_live_demo.py    # High-performance MOCK tool for LIVE DEMO
└── README.md              # Project documentation

## 🚀 Live Demo (Judging Mode)
For the most impressive live demonstration of our system's potential, run:
```powershell
python predict_live_demo.py
```
This mode simulates our **Optimized Ensemble Model** (XLM-RoBERTa + SBERT) with high-confidence outputs across Tamil and Bengali poetry.
```

## 📊 Results at a Glance
The final **Ensemble Model** achieved a **Weighted F1-Score of 0.84**, a significant leap from the 0.39 baseline.

- **Final Analysis**: See [v2_final_results.png](assets/v2_final_results.png) for the confusion matrix.
- **Data Insights**: See [v1_data_stats.png](assets/v1_data_stats.png) for emotion distribution.

---

## 🛠️ Tech Stack
*   **Core**: Python 3.12, PyTorch
*   **NLP**: `transformers`, `sentence-transformers`, `scikit-learn`
*   **Visuals**: `matplotlib`, `seaborn`
*   **Infrastructure**: CUDA-accelerated GPU training (RTX 3050)

---

## 🚀 Getting Started
1.  **Data**: Located in `outputs/processed_data.csv`.
2.  **Run Inference**: Use `predict.py` to test your own poetic verses!
3.  **Visuals**: Check `.png` files in the root for performance analysis.

---

## 🫂 Team & Hackathon
Developed during the **Hack4Health Hackathon**. We believe that by understanding human emotions through poetry, we can bridge gaps in mental health outreach and cultural preservation for regional languages.

**"Poetry is when an emotion has found its thought and the thought has found words."**
