import yaml
import logging
import json
import os
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    model_name = config["models"]["reranker"]["base"]
    save_path = config["models"]["reranker"]["save_path"]
    batch_size = config["models"]["reranker"]["batch_size"]
    epochs = config["models"]["reranker"]["epochs"]
    data_path = os.path.join(config["paths"]["data_dir"], "reranker_pairs.jsonl")
    
    logger.info(f"Loading reranker model: {model_name}")
    model = CrossEncoder(model_name, num_labels=1)
    
    # Load training data from generated pairs
    train_examples = []
    if os.path.exists(data_path):
        with open(data_path) as f:
            for line in f:
                obj = json.loads(line)
                train_examples.append(
                    InputExample(texts=[obj["query"], obj["passage"]], label=float(obj["label"]))
                )
        logger.info(f"Loaded {len(train_examples)} reranker training pairs from {data_path}")
    else:
        logger.warning(f"Data file not found at {data_path}. Run: python -m data.generate_pairs")
        return
    
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    os.makedirs(save_path, exist_ok=True)
    
    logger.info(f"Training reranker for {epochs} epochs on {len(train_examples)} examples...")
    model.fit(
        train_dataloader=train_dataloader,
        epochs=epochs,
        warmup_steps=int(len(train_examples) * 0.1),
        output_path=save_path,
        show_progress_bar=True
    )
    logger.info(f"Reranker fine-tuned and saved to {save_path}")

if __name__ == "__main__":
    main()
