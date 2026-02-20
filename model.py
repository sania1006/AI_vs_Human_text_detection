import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from datasets import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

# LOAD DATA
df = pd.read_csv("AIvsHuman.csv", encoding="latin-1")
print(df.head())
print("Columns in dataset:")
print(df.columns)
exit()
df.dropna(inplace=True)

# SPLIT DATA
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# TOKENIZATION
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=64)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])

train_dataset.set_format("torch")
test_dataset.set_format("torch")

# MODEL
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# TRAINING SETTINGS
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    evaluation_strategy="epoch",
    logging_dir="./logs"
)

# METRICS
def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

# TRAIN
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

# SAVE MODEL
model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model")

print("Model training completed!")