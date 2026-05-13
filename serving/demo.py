import sys
from agent.copilot_agent import SpaceCopilotAgent
import json

def run_demo():
    print("=============================================")
    print("    Space Mission Intelligence Copilot")
    print("=============================================")
    print("Type 'exit' or 'quit' to stop.\n")
    
    try:
        agent = SpaceCopilotAgent()
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        sys.exit(1)

    while True:
        try:
            query = input("\nUser QA > ")
            if query.lower() in ['exit', 'quit']:
                break
            if query.strip() == "":
                continue

            print("\n[Thinking...]")
            response = agent.process_query(query)
            
            print("--- Tool Trace ---")
            for t in response["tool_calls"]:
                print(f"  > Executed: {t['tool']}")
                print(f"    Input:  {json.dumps(t['input'])}")
                print(f"    Output: {str(t['output'])[:100]}...")
                
            print("\n--- Answer ---")
            print(response["answer"])
            
            if response["citations"]:
                print("\n--- Citations ---")
                for c in response["citations"]:
                    print(f"  [{c['doc_id']}] {c['passage'][:100]}...")
            
            if response["escalated"]:
                print(f"\n[!] TICKET ESCALATED: {response['ticket_id']}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error handling request: {e}")

if __name__ == "__main__":
    run_demo()
