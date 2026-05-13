import os
import yaml
import logging
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def load_tool_data(data_path: str):
    queries = []
    labels_str = []
    with open(data_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            queries.append(obj["query"])
            labels_str.append(obj["label"])
            
    unique_labels = sorted(list(set(labels_str)))
    label2id = {l: i for i, l in enumerate(unique_labels)}
    id2label = {i: l for l, i in label2id.items()}
    
    labels = [label2id[l] for l in labels_str]
    
    return Dataset.from_dict({"text": queries, "label": labels}), label2id, id2label

def main():
    config = load_config()
    model_name = config["models"]["tool_policy"]["base"]
    save_path = config["models"]["tool_policy"]["save_path"]
    data_path = os.path.join(config["paths"]["data_dir"], "tool_policy.jsonl")
    
    logger.info(f"Preparing Tool Policy fine-tuning on {model_name}")
    
    dataset, label2id, id2label = load_tool_data(data_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    
    tokenized_dataset = dataset.map(tokenize_fn, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    
    training_args = TrainingArguments(
        output_dir=save_path,
        learning_rate=config["models"]["tool_policy"]["learning_rate"],
        per_device_train_batch_size=config["models"]["tool_policy"]["batch_size"],
        num_train_epochs=config["models"]["tool_policy"]["epochs"],
        logging_steps=10
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    
    logger.info("Starting tool policy training...")
    trainer.train()
    
    logger.info(f"Saving explicitly to {save_path}")
    trainer.save_model(save_path)

if __name__ == "__main__":
    main()
