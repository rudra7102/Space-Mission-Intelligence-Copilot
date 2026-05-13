import unittest
from tools.tools_registry import ToolRegistry

class TestTools(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()

    def test_search_kb(self):
        output = self.registry.execute("SearchKB", {"query": "Mars", "top_k": 1})
        self.assertIn("passages", output)
        self.assertIsInstance(output["passages"], list)

    def test_get_policy(self):
        output = self.registry.execute("GetPolicy", {"section_id": "11A", "agency": "ESA"})
        self.assertEqual(output["section_id"], "11A")
        self.assertEqual(output["agency"], "ESA")
        self.assertIn("policy_text", output)

    def test_create_ticket(self):
        output = self.registry.execute("CreateTicket", {"summary": "Engine Failure", "severity": "High"})
        self.assertEqual(output["status"], "created")
        self.assertIn("TICKET-", output["ticket_id"])
        self.assertEqual(output["eta_hours"], 24)

    def test_compute_success_rate(self):
        output = self.registry.execute("ComputeSuccessRate", {"launch_vehicle": "Falcon 9"})
        self.assertIn("success_rate", output)
        self.assertIsInstance(output["success_rate"], float)

    def test_get_launch_window(self):
        output = self.registry.execute("GetLaunchWindow", {"destination": "Moon"})
        self.assertIn("optimal_windows", output)
        self.assertIsInstance(output["optimal_windows"], list)

if __name__ == '__main__':
    unittest.main()
