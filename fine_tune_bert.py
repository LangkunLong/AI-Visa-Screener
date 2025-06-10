from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import pandas as pd
import ast

MODEL_NAME = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

# using model training csv from bert-large-cased-finetuned-conll03-english
# from huggingface
def load_custom_dataset(csv_file):
    df = pd.read_csv(csv_file)
    df["labels"] = df["labels"].apply(ast.literal_eval) # in case of stringified lists
    df["text"] = df["text"].apply(lambda x: x.split())
    return Dataset.from_pandas(df)

# training, validation, testing datasets
train_dataset = load_custom_dataset("train_data.csv")
val_dataset = load_custom_dataset("val_data.csv")
test_dataset = load_custom_dataset("test_data.csv")


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["text"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        aligned_labels = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)  # ignore
            elif word_idx != previous_word_idx:
                aligned_labels.append(int(label[word_idx]))
            else:
                aligned_labels.append(int(label[word_idx]))
            previous_word_idx = word_idx
        labels.append(aligned_labels)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_train = train_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_val = val_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_test = test_dataset.map(tokenize_and_align_labels, batched=True)

# sample training hyperparmeters
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model("models/visa-ner")