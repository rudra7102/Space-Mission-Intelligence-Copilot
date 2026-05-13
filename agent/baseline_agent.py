"""
baseline_agent.py — A simple RAG baseline agent with NO tool routing and NO preference alignment.
Used for comparison against the full SpaceCopilotAgent to demonstrate quantitative improvement.
"""
import json
from typing import Dict, Any, List
from tools.tools_registry import ToolRegistry
import yaml


class BaselineAgent:
    """
    Baseline: Always uses SearchKB for every query.
    No tool routing, no escalation, no formatting, no relevance checking.
    This represents a naive RAG system.
    """
    def __init__(self):
        with open("config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
        self.registry = ToolRegistry()

    def process_query(self, query: str) -> Dict[str, Any]:
        # Always search KB — no routing intelligence
        tool_output = self.registry.execute("SearchKB", {"query": query, "top_k": 5})
        passages = tool_output.get("passages", [])

        # Raw concatenation of passages — no formatting
        if passages:
            answer = ""
            for p in passages[:3]:
                answer += f"[{p.get('doc_id', '')}] {p.get('text', '')[:200]}\n\n"
        else:
            answer = "No information found."

        # Extract citations (raw doc_ids only)
        citations = []
        for p in passages[:3]:
            citations.append({
                "doc_id": p.get("doc_id", ""),
                "passage": p.get("text", "")[:150]
            })

        return {
            "tool_calls": [{"tool": "SearchKB", "input": {"query": query}, "output": tool_output}],
            "answer": answer,
            "citations": citations,
            "confidence": 0.5,  # No confidence estimation
            "escalated": False,  # Never escalates
            "ticket_id": None,
            "google_url": None
        }
