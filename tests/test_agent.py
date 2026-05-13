import unittest
from tools.tool_loop import ToolExecutionLoop

class TestAgentLoop(unittest.TestCase):
    def setUp(self):
        self.loop = ToolExecutionLoop(max_iterations=2)

    def test_parse_action(self):
        llm_out = "Action: GetPolicy\nAction Input: {\"agency\": \"NASA\"}"
        action, args = self.loop.parse_action(llm_out)
        self.assertEqual(action, "GetPolicy")
        self.assertEqual(args.get("agency"), "NASA")

    def test_escalation_enforcement(self):
        def forced_bad_agent(query, history):
            return "Action: InvalidTool\nAction Input: {}"
            
        result = self.loop.loop(forced_bad_agent, "test query")
        # should max iterations and escalate
        self.assertTrue(result["escalated"])
        self.assertIn("tool_calls", result)

if __name__ == '__main__':
    unittest.main()
