# Space Mission Intelligence Copilot
## Grounded Customer-Support Copilot with Tool Use and Preference Alignment

### DS-615 Neural Networks & Deep Learning — Final Project

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install fastapi uvicorn chromadb pydantic pyyaml pandas sentence-transformers rouge-score
```

### 2. Generate Training Data
```bash
python -m data.generate_pairs
```
This generates 5 data files in `data_storage/`:
- `dpo_pairs.jsonl` (150 DPO preference pairs)
- `tool_policy.jsonl` (600 tool classification samples)
- `retriever_pairs.jsonl` (80+ contrastive pairs)
- `reranker_pairs.jsonl` (60+ relevance pairs)
- `generator_sft.jsonl` (45+ SFT triples)

### 3. Enrich the Knowledge Base
```bash
python -m data.enrich_kb
```
Indexes 15 curated space-domain documents into ChromaDB.

### 4. Start the Demo Server
```bash
python -m uvicorn serving.serve:app --host 0.0.0.0 --port 8000
```
Then open **http://localhost:8000** in your browser.

### 5. Run Evaluation (Baseline vs Copilot)
```bash
python -m evaluation.evaluate
```

### 6. Run Latency Benchmark
```bash
python -m evaluation.benchmark
```

---

## 📁 Project Structure
```
Trial project/
├── agent/
│   ├── copilot_agent.py        # Main agent with tool routing + relevance guard
│   └── baseline_agent.py       # Naive RAG baseline for comparison
├── data/
│   ├── generate_pairs.py       # Training data generator (all 5 datasets)
│   ├── enrich_kb.py            # KB document ingestion (15 documents)
│   ├── build_kb.py             # Vector store builder
│   └── ingest.py               # Data ingestion utilities
├── models/
│   ├── train_retriever.py      # (A) Contrastive retriever (MNR loss)
│   ├── train_reranker.py       # (B) Cross-encoder reranker
│   ├── train_generator.py      # (C) QLoRA Mistral-7B generator
│   ├── train_dpo.py            # (D) DPO preference alignment
│   └── train_tool_policy.py    # (E) BERT tool-policy classifier
├── tools/
│   ├── tools_registry.py       # 5 tools: SearchKB, ComputeSuccessRate, etc.
│   └── tool_loop.py            # ReAct-style tool execution loop
├── evaluation/
│   ├── evaluate.py             # Full evaluation suite (20 queries)
│   ├── metrics.py              # 8 metrics including novel MCA
│   └── benchmark.py            # Latency/throughput benchmarking
├── serving/
│   ├── serve.py                # FastAPI server
│   └── static/
│       ├── index.html          # Glassmorphism UI
│       ├── index.css           # Premium styling
│       └── app.js              # Frontend logic
├── report/
│   ├── main.tex                # 8-page ACL format report
│   └── references.bib          # 11 BibTeX citations
├── config.yaml                 # All hyperparameters
├── requirements.txt            # Python dependencies
├── start_server.bat            # One-click server launch (Windows)
└── README.md                   # This file
```

## 🛠️ Tools Implemented
| Tool | Purpose |
|---|---|
| `SearchKB` | Semantic search over curated space KB |
| `ComputeSuccessRate` | Launch vehicle reliability statistics |
| `GetLaunchWindow` | Orbital mechanics transfer windows |
| `GetPolicy` | Agency regulation lookup (NASA/ESA/ISRO) |
| `CreateTicket` | Escalation for anomalies & KB gaps |

## 📊 Models Trained (5 of 5)
| # | Component | Model | Data Size |
|---|---|---|---|
| A | Retriever | all-MiniLM-L6-v2 | 80+ pairs |
| B | Reranker | cross-encoder/ms-marco | 60+ pairs |
| C | Generator | Mistral-7B (QLoRA) | 45+ triples |
| D | DPO Alignment | Mistral-7B + DPO | 150 pairs |
| E | Tool Policy | BERT-base | 600 samples |

## 📈 Novel Metric: Mission-Criticality Accuracy (MCA)
MCA(p, g) = 1.0 if |p − g| ≤ 0.05, else 0.0

Ensures predicted success rates are within ±5% of verified ground truth.
