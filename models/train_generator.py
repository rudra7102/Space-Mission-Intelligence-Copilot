import os
import yaml
import logging
import json
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    model_name = config["models"]["generator"]["base"]
    save_path = config["models"]["generator"]["save_path"]
    data_path = os.path.join(config["paths"]["data_dir"], "generator_sft.jsonl")
    
    logger.info(f"Preparing QLoRA generator fine-tuning for {model_name}")

    # Load SFT training data from generated file
    sft_texts = []
    if os.path.exists(data_path):
        with open(data_path) as f:
            for line in f:
                obj = json.loads(line)
                sft_texts.append(obj["text"])
        logger.info(f"Loaded {len(sft_texts)} SFT training examples from {data_path}")
    else:
        logger.warning(f"Data file not found at {data_path}. Run: python -m data.generate_pairs")
        return

    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Saving training config for reproducibility.")
        os.makedirs(save_path, exist_ok=True)
        meta = {
            "model": model_name,
            "num_examples": len(sft_texts),
            "lora_rank": config["models"]["generator"]["lora_rank"],
            "lora_alpha": config["models"]["generator"]["lora_alpha"],
            "learning_rate": config["models"]["generator"]["learning_rate"],
            "status": "skipped_no_cuda",
            "note": "QLoRA requires CUDA GPU. Run on a GPU machine to fine-tune."
        }
        with open(os.path.join(save_path, "training_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Training config saved to {save_path}/training_meta.json")
        return

    # Full CUDA training path
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=config["models"]["generator"]["lora_rank"],
        lora_alpha=config["models"]["generator"]["lora_alpha"],
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    dataset = Dataset.from_dict({"text": sft_texts})

    training_args = TrainingArguments(
        output_dir=save_path,
        per_device_train_batch_size=config["models"]["generator"]["batch_size"],
        gradient_accumulation_steps=4,
        learning_rate=config["models"]["generator"]["learning_rate"],
        logging_steps=10,
        num_train_epochs=config["models"]["generator"]["epochs"],
        optim="paged_adamw_8bit",
        save_steps=50,
        warmup_steps=10,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        args=training_args,
    )
    
    logger.info(f"Starting QLoRA SFT tuning on {len(sft_texts)} examples...")
    trainer.train()
    
    logger.info(f"Saving PEFT adapter to {save_path}")
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    main()
