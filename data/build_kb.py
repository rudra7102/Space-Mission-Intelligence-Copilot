import os
import uuid
import yaml
import json
import logging
import pandas as pd
from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        i += (chunk_size - overlap)
    return chunks

def build_documents(data_dir: str) -> List[Dict[str, Any]]:
    docs = []
    
    # 1. SpaceX processing
    spacex_file = os.path.join(data_dir, "spacex_launches.json")
    if os.path.exists(spacex_file):
        with open(spacex_file, "r") as f:
            data = json.load(f)
            for launch in data:
                text = f"SpaceX launch {launch.get('name', 'N/A')} occurred on {launch.get('date_utc', 'N/A')}. Success: {launch.get('success', False)}."
                docs.extend([{"text": c, "metadata": {"source": "spacex", "title": launch.get('name', 'N/A')}} for c in chunk_text(text)])

    # 2. Exoplanet processing
    exo_file = os.path.join(data_dir, "exoplanets.csv")
    if os.path.exists(exo_file):
        df = pd.read_csv(exo_file)
        for _, row in df.iterrows():
            text = f"Exoplanet {row.get('pl_name', '')} orbits {row.get('hostname', '')}. Discovered via {row.get('discoverymethod', '')}."
            docs.extend([{"text": c, "metadata": {"source": "nasa_exoplanet", "title": str(row.get('pl_name', ''))}} for c in chunk_text(text)])
            
    # Include synthetic policies 
    policies = [
        "NASA Safety Policy section 104A: All lunar missions must ensure a secondary life-support redundancy.",
        "ESA Orbital Debris limitation guidelines: Satellites in LEO must de-orbit within 25 years of EOL."
    ]
    for p in policies:
        docs.extend([{"text": c, "metadata": {"source": "policy", "title": "Space Agency Policy"}} for c in chunk_text(p)])

    return docs

def main():
    config = load_config()
    data_dir = config["paths"]["data_dir"]
    kb_dir = config["paths"]["kb_dir"]
    chunk_size = config["data_pipeline"]["chunk_size"]
    chunk_overlap = config["data_pipeline"]["chunk_overlap"]
    model_name = config["models"]["retriever"]["base"]
    
    os.makedirs(kb_dir, exist_ok=True)
    
    logger.info("Initializing vector DB...")
    client = chromadb.PersistentClient(path=kb_dir)
    collection = client.get_or_create_collection(name="space_kb")
    
    logger.info("Building raw documents...")
    docs = build_documents(data_dir)
    
    logger.info(f"Generated {len(docs)} document chunks. Loading embedder: {model_name}")
    embedder = SentenceTransformer(model_name)
    
    # Batch embedding directly to collection
    batch_size = 64
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        texts = [b["text"] for b in batch]
        metas = [b["metadata"] for b in batch]
        ids = [str(uuid.uuid4()) for _ in batch]
        
        embeddings = embedder.encode(texts).tolist()
        
        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metas,
            ids=ids
        )
        logger.info(f"Embedded batch {i//batch_size + 1}")

    logger.info(f"KB successfully built at {kb_dir} with {collection.count()} items.")

if __name__ == "__main__":
    main()
