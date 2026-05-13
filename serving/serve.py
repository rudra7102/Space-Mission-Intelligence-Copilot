import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import yaml
import time
from typing import List, Dict, Any, Optional
from agent.copilot_agent import SpaceCopilotAgent

app = FastAPI(title="Space Mission Intelligence Copilot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static directory exists
os.makedirs("serving/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="serving/static"), name="static")

@app.get("/")
def read_index():
    return FileResponse("serving/static/index.html")

# Setup agent globally
try:
    agent = SpaceCopilotAgent()
    kb_size = 500 # mocked, could be fetched via agent.tool_loop.registry.collection.count()
except Exception as e:
    agent = None
    kb_size = 0

class QueryRequest(BaseModel):
    question: str
    history: List[Dict[str, str]] = []
    user_id: str = "default_user"

class QueryResponse(BaseModel):
    tool_calls: List[Dict[str, Any]]
    answer: str
    citations: List[Dict[str, str]]
    confidence: float
    escalated: bool
    ticket_id: Optional[str]
    google_url: Optional[str]
    latency_ms: float

@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    if not agent:
        raise HTTPException(status_code=500, detail="Agent is not initialized correctly.")
    
    start_time = time.time()
    result = agent.process_query(req.question)
    latency = (time.time() - start_time) * 1000.0
    
    return QueryResponse(
        tool_calls=result["tool_calls"],
        answer=result["answer"],
        citations=result["citations"],
        confidence=result["confidence"],
        escalated=result["escalated"],
        ticket_id=result["ticket_id"],
        google_url=result.get("google_url"),
        latency_ms=latency
    )

@app.get("/health")
def health_endpoint():
    return {
        "status": "ok",
        "model": "mistralai/Mistral-7B-Instruct-v0.2-QLoRA-DPO",
        "kb_size": kb_size
    }

def run_server():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    print("Starting FastAPI Server...")
    uvicorn.run(app, host=config["server"]["host"], port=config["server"]["port"])

if __name__ == "__main__":
    run_server()
