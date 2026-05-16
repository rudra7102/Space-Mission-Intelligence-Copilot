# Space Mission Intelligence Copilot

## Grounded AI Copilot for Mission-Critical Aerospace Support

Space Mission Intelligence Copilot is a grounded AI system designed for aerospace support workflows using Retrieval-Augmented Generation (RAG), semantic retrieval, tool-augmented reasoning, and preference-aligned response generation.

The project addresses key limitations of general-purpose Large Language Models (LLMs) in mission-critical environments, including hallucinated outputs, lack of source grounding, unsafe responses, and inability to escalate low-confidence scenarios reliably.

---

# Live Demo

https://space-mission-intelligence-copilot.onrender.com/

---

# Overview

The system combines:

* Retrieval-Augmented Generation (RAG)
* Tool-aware reasoning
* QLoRA fine-tuning
* Direct Preference Optimization (DPO)
* Semantic retrieval with vector databases
* Hallucination prevention mechanisms
* Human escalation workflows

The architecture is designed to generate grounded, citation-aware, and reliable responses for aerospace-related queries.

---

# Key Features

* Semantic retrieval over curated aerospace knowledge bases
* Deterministic tool routing for specialized query handling
* Citation-first grounded response generation
* Knowledge-base relevance validation
* Hallucination prevention using relevance guards
* Human escalation support for low-confidence queries
* FastAPI-based deployment architecture
* Interactive frontend interface

---

# System Architecture

```text
User Query
   ↓
Space Domain Guard
   ↓
Deterministic Tool Router
   ↓
Tool Execution Layer
   ↓
KB Relevance Guard
   ↓
Grounded Response + Citations
```

---

# Tools Implemented

| Tool               | Purpose                                       |
| ------------------ | --------------------------------------------- |
| SearchKB           | Semantic search over aerospace knowledge base |
| ComputeSuccessRate | Launch reliability statistics                 |
| GetLaunchWindow    | Orbital transfer and launch window queries    |
| GetPolicy          | NASA / ESA / ISRO policy retrieval            |
| CreateTicket       | Escalation support for unresolved queries     |

---

# Models and Components

| Component              | Model                 | Purpose                      |
| ---------------------- | --------------------- | ---------------------------- |
| Retriever              | all-MiniLM-L6-v2      | Semantic retrieval           |
| Reranker               | CrossEncoder/ms-marco | Relevance ranking            |
| Generator              | Mistral-7B + QLoRA    | Grounded response generation |
| DPO Alignment          | Mistral-7B + DPO      | Preference optimization      |
| Tool Policy Classifier | BERT-base             | Tool routing classification  |

---

# Training Pipeline

The project includes custom-generated domain-specific datasets for multiple learning objectives.

## Generated Datasets

* 150 DPO preference pairs
* 600 tool-policy classification samples
* 80+ retriever contrastive pairs
* 60+ reranker relevance pairs
* 45+ supervised fine-tuning triples

---

# Data Sources

The knowledge base and datasets were curated using verified aerospace sources, including:

* SpaceX REST API
* NASA Exoplanet Archive
* ESA policy documents
* ISRO mission documentation
* Curated orbital mechanics references

---

# Evaluation Highlights

* +85% Tool Selection Accuracy
* 100% Escalation Recall
* Improved grounding and citation quality
* Sub-50ms average latency

---

# Custom Evaluation Metric

## Mission-Criticality Accuracy (MCA)

```text
MCA(p, g) = 1.0 if |p − g| ≤ 0.05, else 0.0
```

This metric validates whether predicted mission statistics remain within ±5% of verified ground-truth values.

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/rudra7102/Space-Mission-Intelligence-Copilot.git
cd Space-Mission-Intelligence-Copilot
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## 1. Generate Training Data

```bash
python -m data.generate_pairs
```

This generates:

* DPO preference datasets
* Tool-policy datasets
* Retriever training pairs
* Reranker datasets
* Generator fine-tuning datasets

---

## 2. Build and Enrich Knowledge Base

```bash
python -m data.enrich_kb
```

This indexes curated aerospace documents into ChromaDB.

---

## 3. Start the FastAPI Server

```bash
python -m uvicorn serving.serve:app --host 0.0.0.0 --port 8000
```

Open the application in browser:

```text
http://localhost:8000
```

---

# Evaluation

## Run Evaluation Suite

```bash
python -m evaluation.evaluate
```

## Run Latency Benchmark

```bash
python -m evaluation.benchmark
```

---

# Project Structure

```text
agent/         → AI agent pipeline
data/          → dataset generation & ingestion
models/        → training scripts
tools/         → tool execution framework
evaluation/    → evaluation & benchmarks
serving/       → FastAPI backend & frontend
report/        → research report & references
```

---

# Technologies Used

* Python
* FastAPI
* HuggingFace Transformers
* Sentence Transformers
* ChromaDB
* PyTorch
* QLoRA
* DPO
* BERT
* Vector Databases
* Retrieval-Augmented Generation (RAG)

---

# Research Areas Explored

* Retrieval-Augmented Generation
* Tool-Augmented AI Systems
* LLM Alignment
* Preference Optimization
* Semantic Retrieval
* Hallucination Prevention
* Grounded AI Systems
* Vector Search Systems

---

# ============================================================
#  SpaceCopilot — 20 Test Questions (Evaluation Suite)
#  Use these to demo and test all core system behaviors
# ============================================================


# ─────────────────────────────────────────────────────────────
# GROUP 1 — SearchKB
# Expected:
# - Semantic retrieval from ChromaDB
# - Grounded aerospace responses
# - Passage cards with doc_id citations
# ─────────────────────────────────────────────────────────────

Q1:  Tell me about the James Webb Space Telescope
Q2:  What is the International Space Station?
Q3:  Describe the Hubble Space Telescope mission
Q4:  Tell me about SpaceX Starlink constellation
Q5:  What satellites does ESA operate?


# ─────────────────────────────────────────────────────────────
# GROUP 2 — ComputeSuccessRate
# Expected:
# - Reliability statistics
# - Mission success/failure analysis
# - Structured numerical outputs
# ─────────────────────────────────────────────────────────────

Q6:  What is the success rate of Falcon 9?
Q7:  What is the success rate of Falcon Heavy?
Q8:  Calculate the reliability of Soyuz launches
Q9:  What is the historical failure probability of Ariane 5?


# ─────────────────────────────────────────────────────────────
# GROUP 3 — GetLaunchWindow
# Expected:
# - Launch window recommendations
# - Delta-v estimates
# - Orbital transfer analysis
# ─────────────────────────────────────────────────────────────

Q10: What is the next optimal launch window to Mars?
Q11: When should we launch a mission to the Moon?
Q12: Best time to launch to Jupiter?


# ─────────────────────────────────────────────────────────────
# GROUP 4 — GetPolicy
# Expected:
# - Agency policy retrieval
# - Citation-grounded policy responses
# - Last-updated metadata
# ─────────────────────────────────────────────────────────────

Q13: Is there a NASA policy regarding orbital debris for satellites in LEO?
Q14: What are ESA guidelines for satellite end-of-life disposal?
Q15: What is ISRO's policy on debris mitigation?


# ─────────────────────────────────────────────────────────────
# GROUP 5 — CreateTicket
# Expected:
# - Human escalation workflow
# - Ticket creation
# - Mission-critical alert handling
# ─────────────────────────────────────────────────────────────

Q16: I need to report a critical anomaly in my satellite's attitude control system.
Q17: Our propulsion module has stopped working. This is mission-critical.
Q18: There is a critical temperature anomaly in the thermal management unit.


# ─────────────────────────────────────────────────────────────
# GROUP 6 — Out-of-Scope Guard
# Expected:
# - Query rejection
# - No tool execution
# - Safe fallback behavior
# ─────────────────────────────────────────────────────────────

Q19: what is maths?
Q20: abcd


# ============================================================
# EXPECTED OUTPUT SUMMARY
# ============================================================
#
# Q1-Q5
# → SearchKB
# → Grounded aerospace passages + document citations
#
# Q6
# → ComputeSuccessRate
# → Falcon 9: 98.4%, 229 missions
#
# Q7
# → ComputeSuccessRate
# → Falcon Heavy: 93.3%, 9 missions
#
# Q8
# → ComputeSuccessRate
# → Soyuz: 97.9%, 142 missions
#
# Q9
# → ComputeSuccessRate
# → Ariane 5: 97.4%, 117 missions
#
# Q10
# → GetLaunchWindow
# → 2026-10-15, 2028-11-20, Δv 4300 m/s
#
# Q11
# → GetLaunchWindow
# → 2026-06-15, 2027-01-10, Δv 3200 m/s
#
# Q12
# → GetLaunchWindow
# → 2028-03-20, 2030-09-15, Δv 6100 m/s
#
# Q13
# → GetPolicy
# → NASA-STD-8719.14B, 25-year deorbit rule
#
# Q14
# → GetPolicy
# → ESA ESSB-ST-U-007, passivation guidelines
#
# Q15
# → GetPolicy
# → ISRO SDMP-01, IADC-aligned debris mitigation
#
# Q16-Q18
# → CreateTicket
# → TICKET-XXXXXXXX + escalation workflow
#
# Q19-Q20
# → Out-of-Scope Guard
# → 🚫 Query blocked safely with no tool call
#
# ============================================================
# BONUS TEST — KB Gap Handling
# ============================================================
#
# EXTRA:
# Tell me about the Chandrayaan-3 lunar landing
#
# Expected:
# - KB relevance failure detected
# - External search recommendation
# - Optional escalation workflow
#
# ============================================================

# Author

Rudra Pandit

---


# Links

## Live Demo

https://space-mission-intelligence-copilot.onrender.com/

