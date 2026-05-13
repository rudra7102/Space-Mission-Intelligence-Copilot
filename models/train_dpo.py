import os
import yaml
import logging
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import PeftModel
from trl import DPOTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def load_dpo_data(data_path: str):
    prompts = []
    chosens = []
    rejecteds = []
    with open(data_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            prompts.append(obj["query"])
            chosens.append(obj["chosen"])
            rejecteds.append(obj["rejected"])
            
    return Dataset.from_dict({
        "prompt": prompts,
        "chosen": chosens,
        "rejected": rejecteds
    })

def main():
    config = load_config()
    model_name = config["models"]["generator"]["base"]
    lora_path = config["models"]["generator"]["save_path"]
    save_path = config["models"]["dpo"]["save_path"]
    dpo_data_path = os.path.join(config["paths"]["data_dir"], "dpo_pairs.jsonl")
    
    logger.info("Initializing DPO Fine-tuning on top of QLoRA generator.")
    
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Skipping DPO training in mock local run.")
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "mock_success.txt"), "w") as f:
            f.write("DPO mock success!")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model, then apply peft wrapper
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model = PeftModel.from_pretrained(model, lora_path, is_trainable=True)
    
    # Needs a reference model for DPO
    ref_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    ref_model = PeftModel.from_pretrained(ref_model, lora_path)
    
    train_dataset = load_dpo_data(dpo_data_path)
    
    training_args = TrainingArguments(
        output_dir=save_path,
        per_device_train_batch_size=config["models"]["dpo"]["batch_size"],
        learning_rate=config["models"]["dpo"]["learning_rate"],
        logging_steps=10,
        max_steps=100
    )
    
    dpo_trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        beta=config["models"]["dpo"]["beta"],
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    
    logger.info("Starting DPO training...")
    dpo_trainer.train()
    
    logger.info(f"Saving DPO aligned model to {save_path}")
    dpo_trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    main()
