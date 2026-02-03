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
    
    # Class Weights
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
