import json
from typing import List, Dict, Any, Tuple
from tools.tools_registry import ToolRegistry
import logging

logger = logging.getLogger(__name__)

class ToolExecutionLoop:
    def __init__(self, max_iterations: int = 4):
        self.registry = ToolRegistry()
        self.max_iterations = max_iterations

    def parse_action(self, llm_output: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parses LLM text to extract ToolName and JSON args.
        Expects format like:
        Action: SearchKB
        Action Input: {"query": "Apollo 11"}
        """
        try:
            lines = llm_output.split('\n')
            action = None
            action_input_str = ""
            for line in lines:
                if line.startswith("Action:"):
                    action = line.replace("Action:", "").strip()
                elif line.startswith("Action Input:"):
                    action_input_str = line.replace("Action Input:", "").strip()
            
            if action and action_input_str:
                return action, json.loads(action_input_str)
        except Exception as e:
            logger.warning(f"Failed to parse action: {e}")
        return None, {}

    def loop(self, agent_step_fn, query: str) -> Dict[str, Any]:
        """
        Executes ReAct style loop.
        agent_step_fn: (query, history) -> str (LLM response block)
        """
        history = []
        tool_calls_trace = []
        escalated = False
        ticket_id = None
        
        for i in range(self.max_iterations):
            llm_response = agent_step_fn(query, history)
            history.append(f"Thought/Action: {llm_response}")
            
            action_name, action_args = self.parse_action(llm_response)
            
            if not action_name:
                # LLM chose to output Final Answer
                return {
                    "tool_calls": tool_calls_trace,
                    "answer": llm_response,
                    "citations": [], # We'll extract this later
                    "confidence": 0.85,
                    "escalated": escalated,
                    "ticket_id": ticket_id
                }
                
            output = self.registry.execute(action_name, action_args)
            tool_calls_trace.append({"tool": action_name, "input": action_args, "output": output})
            
            history.append(f"Observation: {json.dumps(output)}")
            
            if action_name == "CreateTicket":
                escalated = True
                ticket_id = output.get("ticket_id")
                # Immediately break and output if escalated
                return {
                    "tool_calls": tool_calls_trace,
                    "answer": "I have escalated your issue.",
                    "citations": [],
                    "confidence": 1.0,
                    "escalated": escalated,
                    "ticket_id": ticket_id
                }

        # If we reach here, we exhausted iterations. Mandate escalation.
        es_out = self.registry.execute("CreateTicket", {
            "summary": "Auto-escalation due to context iteration exhaustion", 
            "mission_context": query
        })
        tool_calls_trace.append({"tool": "CreateTicket", "input": {}, "output": es_out})
        return {
            "tool_calls": tool_calls_trace,
            "answer": "I could not find the answer in the KB. Esculated.",
            "citations": [],
            "confidence": 0.0,
            "escalated": True,
            "ticket_id": es_out.get("ticket_id")
        }
