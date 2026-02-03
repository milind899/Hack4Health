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

## 📊 Results at a Glance

| Metric | Baseline | **Our Optimized System** |
| :--- | :--- | :--- |
| **Weighted F1-Score** | 0.39 | **0.84** |
| **Handling 40+ Emotions**| Struggles | **Robust** |
| **Language Support** | Keyword-based | **Semantic-based** |

> [!TIP]
> **Check out our Visuals!**
> We have included high-fidelity confusion matrices and architecture diagrams in the root folder to show exactly how the model performs.

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
