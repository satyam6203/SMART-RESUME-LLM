import os
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Get project root directory (parent of 'training/')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# Load dataset
dataset_path = os.path.join(BASE_DIR, "dataset", "resume_dataset.json")
with open(dataset_path, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = []

for item in data:
    text = f"Input: {item['input']}\nOutput: {item['output']}"
    texts.append({"text": text})

dataset = Dataset.from_list(texts)

def tokenize(example):
    tokens = tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset.map(tokenize)

output_dir = os.path.join(BASE_DIR, "model", "resume_llm")

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_steps=500,
    logging_steps=1,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Training complete! Model saved to {output_dir}")