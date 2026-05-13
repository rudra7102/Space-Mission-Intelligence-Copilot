"""
evaluate.py — Full evaluation suite that:
1. Runs the same test set against BOTH baseline and full copilot
2. Computes real metrics (Citation-F1, Answer-in-KB, Tool Accuracy, Escalation, MCA)
3. Outputs a comparison table + saves results to JSON for the report
"""
import logging
import json
import time
import os
from evaluation.metrics import MetricEvaluator
from agent.copilot_agent import SpaceCopilotAgent
from agent.baseline_agent import BaselineAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Comprehensive test set covering ALL tools ──
TEST_SET = [
    # SearchKB queries
    {"query": "Tell me about the James Webb Space Telescope",
     "gold_tool": "SearchKB", "gold_escalate": False, "true_sr": 0.0,
     "expected_keywords": ["jwst", "webb", "infrared", "l2", "mirror"]},

    {"query": "What is the International Space Station?",
     "gold_tool": "SearchKB", "gold_escalate": False, "true_sr": 0.0,
     "expected_keywords": ["iss", "station", "orbit", "leo", "408"]},

    {"query": "Describe the Hubble Space Telescope mission",
     "gold_tool": "SearchKB", "gold_escalate": False, "true_sr": 0.0,
     "expected_keywords": ["hubble", "telescope", "mirror", "servicing"]},

    {"query": "Tell me about SpaceX Starlink constellation",
     "gold_tool": "SearchKB", "gold_escalate": False, "true_sr": 0.0,
     "expected_keywords": ["starlink", "satellite", "broadband", "leo"]},

    {"query": "What satellites does ESA operate?",
     "gold_tool": "SearchKB", "gold_escalate": False, "true_sr": 0.0,
     "expected_keywords": ["esa", "sentinel", "copernicus", "gaia"]},

    # ComputeSuccessRate queries
    {"query": "What is the success rate of Falcon 9?",
     "gold_tool": "ComputeSuccessRate", "gold_escalate": False, "true_sr": 0.984,
     "expected_keywords": ["98", "229", "falcon"]},

    {"query": "What is the success rate of Falcon Heavy?",
     "gold_tool": "ComputeSuccessRate", "gold_escalate": False, "true_sr": 0.933,
     "expected_keywords": ["93", "9", "falcon heavy"]},

    {"query": "Calculate the reliability of Soyuz launches",
     "gold_tool": "ComputeSuccessRate", "gold_escalate": False, "true_sr": 0.979,
     "expected_keywords": ["97", "142", "soyuz"]},

    {"query": "What is the historical failure probability of Ariane 5?",
     "gold_tool": "ComputeSuccessRate", "gold_escalate": False, "true_sr": 0.974,
     "expected_keywords": ["97", "117", "ariane"]},

    # GetLaunchWindow queries
    {"query": "What is the next optimal launch window to Mars?",
     "gold_tool": "GetLaunchWindow", "gold_escalate": False, "true_sr": 0.0,
     "expected_keywords": ["2026", "2028", "mars", "delta"]},

    {"query": "When should we launch to the Moon?",
     "gold_tool": "GetLaunchWindow", "gold_escalate": False, "true_sr": 0.0,
     "expected_keywords": ["2026", "2027", "moon", "lunar"]},

    {"query": "Best time to launch to Jupiter?",
     "gold_tool": "GetLaunchWindow", "gold_escalate": False, "true_sr": 0.0,
     "expected_keywords": ["2028", "2030", "jupiter"]},

    # GetPolicy queries
    {"query": "Is there a NASA policy regarding orbital debris for satellites in LEO?",
     "gold_tool": "GetPolicy", "gold_escalate": False, "true_sr": 0.0,
     "expected_keywords": ["nasa", "debris", "25", "deorbit", "8719"]},

    {"query": "What are ESA guidelines for satellite end-of-life disposal?",
     "gold_tool": "GetPolicy", "gold_escalate": False, "true_sr": 0.0,
     "expected_keywords": ["esa", "debris", "passivat", "end-of-life"]},

    {"query": "What is ISRO's policy on debris mitigation?",
     "gold_tool": "GetPolicy", "gold_escalate": False, "true_sr": 0.0,
     "expected_keywords": ["isro", "iadc", "25"]},

    # CreateTicket queries
    {"query": "I need to report a critical anomaly in my satellite's attitude control system.",
     "gold_tool": "CreateTicket", "gold_escalate": True, "true_sr": 0.0,
     "expected_keywords": ["ticket", "escalat"]},

    {"query": "Our propulsion module has stopped working. This is mission-critical.",
     "gold_tool": "CreateTicket", "gold_escalate": True, "true_sr": 0.0,
     "expected_keywords": ["ticket", "escalat"]},

    {"query": "There is a critical temperature anomaly in the thermal management unit.",
     "gold_tool": "CreateTicket", "gold_escalate": True, "true_sr": 0.0,
     "expected_keywords": ["ticket", "escalat"]},

    # Out-of-scope queries (should NOT escalate or return noise)
    {"query": "what is maths?",
     "gold_tool": "OutOfScope", "gold_escalate": False, "true_sr": 0.0,
     "expected_keywords": ["scope", "space"]},

    {"query": "abcd",
     "gold_tool": "OutOfScope", "gold_escalate": False, "true_sr": 0.0,
     "expected_keywords": ["scope"]},
]


def evaluate_agent(agent, agent_name: str, evaluator: MetricEvaluator):
    """Run full evaluation on an agent and return metrics dict."""
    logger.info(f"\n{'='*50}")
    logger.info(f"  Evaluating: {agent_name}")
    logger.info(f"{'='*50}")

    results = {
        "agent": agent_name,
        "total_queries": len(TEST_SET),
        "citation_f1_scores": [],
        "grounding_scores": [],
        "tool_accuracy_hits": 0,
        "tool_call_successes": 0,
        "tool_call_total": 0,
        "escalation_tp": 0, "escalation_fp": 0, "escalation_fn": 0,
        "mca_scores": [],
        "mca_counted": 0,
        "latencies_ms": [],
        "keyword_hit_scores": [],
    }

    for idx, item in enumerate(TEST_SET):
        start = time.time()
        try:
            out = agent.process_query(item["query"])
        except Exception as e:
            logger.warning(f"  Query {idx} failed: {e}")
            results["latencies_ms"].append(0)
            continue
        latency = (time.time() - start) * 1000
        results["latencies_ms"].append(latency)

        answer = out.get("answer", "")
        tool_calls = out.get("tool_calls", [])

        # ── M1: Citation-F1 (check if doc_ids appear in answer) ──
        cited_ids = []
        for c in out.get("citations", []):
            if c.get("doc_id"):
                cited_ids.append(c["doc_id"])
        # Credit if any citation exists in answer text
        if cited_ids:
            found = sum(1 for cid in cited_ids if cid[:8] in answer)
            cf1 = found / len(cited_ids) if cited_ids else 0
        else:
            cf1 = 0.0
        results["citation_f1_scores"].append(cf1)

        # ── M2: Keyword grounding (do expected keywords appear?) ──
        expected_kw = item.get("expected_keywords", [])
        if expected_kw:
            kw_hits = sum(1 for kw in expected_kw if kw.lower() in answer.lower())
            kw_score = kw_hits / len(expected_kw)
        else:
            kw_score = 0.0
        results["keyword_hit_scores"].append(kw_score)

        # ── M3: Grounding / Answer-in-KB overlap ──
        passages = []
        for tc in tool_calls:
            if tc.get("tool") == "SearchKB":
                for p in tc.get("output", {}).get("passages", []):
                    passages.append(p.get("text", ""))
        grounding = evaluator.answer_in_kb_rate(answer, passages) if passages else 0.0
        results["grounding_scores"].append(grounding)

        # ── M4: Tool Selection Accuracy ──
        if tool_calls:
            pred_tool = tool_calls[0]["tool"]
        elif "Out of Scope" in answer or "scope" in answer.lower():
            pred_tool = "OutOfScope"
        else:
            pred_tool = "None"
        
        if pred_tool == item["gold_tool"]:
            results["tool_accuracy_hits"] += 1

        # ── M5: Tool Call Success Rate ──
        if tool_calls:
            results["tool_call_total"] += len(tool_calls)
            for tc in tool_calls:
                if "error" not in tc.get("output", {}):
                    results["tool_call_successes"] += 1

        # ── M6: Escalation precision/recall ──
        pred_escalate = out.get("escalated", False)
        gold_escalate = item["gold_escalate"]
        if pred_escalate and gold_escalate:
            results["escalation_tp"] += 1
        elif pred_escalate and not gold_escalate:
            results["escalation_fp"] += 1
        elif not pred_escalate and gold_escalate:
            results["escalation_fn"] += 1

        # ── M7: Mission-Criticality Accuracy ──
        if item["true_sr"] > 0 and tool_calls:
            for tc in tool_calls:
                if tc.get("tool") == "ComputeSuccessRate":
                    pred_sr = tc.get("output", {}).get("success_rate", 0.0)
                    mca = evaluator.mission_criticality_accuracy(pred_sr, item["true_sr"])
                    results["mca_scores"].append(mca)
                    results["mca_counted"] += 1

        logger.info(f"  [{idx+1:2d}/{len(TEST_SET)}] {item['query'][:50]:50s} | Tool: {pred_tool:20s} | ✓={pred_tool==item['gold_tool']} | {latency:.0f}ms")

    # ── Aggregate metrics ──
    n = len(TEST_SET)
    metrics = {
        "agent": agent_name,
        "citation_f1": sum(results["citation_f1_scores"]) / n if n else 0,
        "keyword_grounding": sum(results["keyword_hit_scores"]) / n if n else 0,
        "answer_in_kb_rate": sum(results["grounding_scores"]) / max(1, len([s for s in results["grounding_scores"] if s > 0])),
        "tool_selection_accuracy": results["tool_accuracy_hits"] / n if n else 0,
        "tool_call_success_rate": results["tool_call_successes"] / max(1, results["tool_call_total"]),
        "escalation_precision": results["escalation_tp"] / max(1, results["escalation_tp"] + results["escalation_fp"]),
        "escalation_recall": results["escalation_tp"] / max(1, results["escalation_tp"] + results["escalation_fn"]),
        "mca": sum(results["mca_scores"]) / max(1, results["mca_counted"]),
        "avg_latency_ms": sum(results["latencies_ms"]) / max(1, len(results["latencies_ms"])),
        "p95_latency_ms": sorted(results["latencies_ms"])[int(0.95 * len(results["latencies_ms"]))] if results["latencies_ms"] else 0,
    }
    return metrics


def print_comparison(baseline_m, copilot_m):
    """Print side-by-side comparison table."""
    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS — Baseline vs Full Copilot")
    print("=" * 70)
    print(f"{'Metric':<30s} {'Baseline':>12s} {'Full Copilot':>14s} {'Δ':>8s}")
    print("-" * 70)

    rows = [
        ("Citation-F1", "citation_f1"),
        ("Keyword Grounding", "keyword_grounding"),
        ("Answer-in-KB Rate", "answer_in_kb_rate"),
        ("Tool Selection Accuracy", "tool_selection_accuracy"),
        ("Tool Call Success Rate", "tool_call_success_rate"),
        ("Escalation Precision", "escalation_precision"),
        ("Escalation Recall", "escalation_recall"),
        ("Mission-Crit. Accuracy (MCA)", "mca"),
        ("Avg Latency (ms)", "avg_latency_ms"),
        ("P95 Latency (ms)", "p95_latency_ms"),
    ]

    for label, key in rows:
        bv = baseline_m.get(key, 0)
        cv = copilot_m.get(key, 0)
        delta = cv - bv
        if key.endswith("_ms"):
            print(f"  {label:<28s} {bv:>10.1f}ms {cv:>12.1f}ms {delta:>+7.1f}")
        else:
            print(f"  {label:<28s} {bv:>11.2f} {cv:>13.2f} {delta:>+7.2f}")
    print("=" * 70)


def evaluate_run():
    logger.info("=" * 55)
    logger.info("  Space Mission Copilot — Full Evaluation Suite")
    logger.info("=" * 55)

    evaluator = MetricEvaluator()

    # Initialize both agents
    logger.info("Initializing Full Copilot Agent...")
    copilot = SpaceCopilotAgent()

    logger.info("Initializing Baseline Agent...")
    baseline = BaselineAgent()

    # Run evaluation on both
    baseline_metrics = evaluate_agent(baseline, "Baseline (Raw RAG)", evaluator)
    copilot_metrics = evaluate_agent(copilot, "Full Copilot", evaluator)

    # Print comparison
    print_comparison(baseline_metrics, copilot_metrics)

    # Save results to JSON
    output = {
        "baseline": baseline_metrics,
        "copilot": copilot_metrics,
        "test_set_size": len(TEST_SET),
    }
    os.makedirs("evaluation", exist_ok=True)
    out_path = os.path.join("evaluation", "results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    evaluate_run()
