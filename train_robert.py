import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

# 1. LOAD DATASET
CSV_FILE = "AIvsHuman.csv"

df = pd.read_csv(CSV_FILE,encoding="latin1")

texts = df["text"].astype(str).tolist()
labels = df["generated"].astype(int).tolist()

print("Total samples:", len(texts))

# 2. DATASET CLASS

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=256
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 3. TOKENIZER + DATASET

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

full_dataset = TextDataset(texts, labels, tokenizer)

# Train / Validation split (90 / 10)
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

print("Training samples:", train_size)
print("Validation samples:", val_size)

# 4. LOAD MODEL

model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=2
)
# 5. TRAINING SETTINGS (CPU SAFE)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
# 6. TRAIN MODEL

trainer.train()

# 7. SAVE MODEL

model.save_pretrained("ai_human_detector")
tokenizer.save_pretrained("ai_human_detector")

print(" Model saved in folder: ai_human_detector")

# 8. TEST WITH USER INPUT

print("\nEnter text to test (type 'exit' to stop)\n")

model.eval()

while True:
    user_text = input("Text: ")

    if user_text.lower() == "exit":
        break

    inputs = tokenizer(
        user_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    if prediction == 1:
        print("Prediction → HUMAN TEXT\n")
    else:
        print("Prediction → AI GENERATED TEXT\n")