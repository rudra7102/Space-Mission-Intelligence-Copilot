import re
from typing import List, Dict, Any
from rouge_score import rouge_scorer

class MetricEvaluator:
    """Comprehensive metrics evaluator for the Space Copilot system."""

    def __init__(self):
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def citation_f1(self, answer: str, expected_citations: List[str]) -> float:
        """Citation-F1: fraction of expected citations present in the answer text."""
        if not expected_citations:
            return 1.0
        found = sum(1 for cp in expected_citations if cp in answer)
        return found / len(expected_citations)

    def answer_in_kb_rate(self, answer: str, passages: List[str]) -> float:
        """Token-level F1 overlap between answer and best retrieved passage."""
        if not passages:
            return 0.0

        def f1(a, p):
            a_toks = set(a.lower().split())
            p_toks = set(p.lower().split())
            if not a_toks or not p_toks:
                return 0.0
            overlap = a_toks.intersection(p_toks)
            prec = len(overlap) / len(a_toks) if a_toks else 0
            rec = len(overlap) / len(p_toks) if p_toks else 0
            return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        return max(f1(answer, p) for p in passages)

    def keyword_grounding_score(self, answer: str, expected_keywords: List[str]) -> float:
        """Fraction of expected domain keywords present in the answer."""
        if not expected_keywords:
            return 1.0
        found = sum(1 for kw in expected_keywords if kw.lower() in answer.lower())
        return found / len(expected_keywords)

    def faithfulness_score(self, answer: str, passages: List[str]) -> float:
        """
        Faithfulness: measures whether the answer is grounded in retrieved passages.
        Uses ROUGE-L between answer and best passage as a proxy for faithfulness.
        Score normalized to 0-5 scale.
        """
        if not passages:
            return 0.0
        max_rouge = 0.0
        for p in passages:
            score = self.rouge.score(p, answer)['rougeL'].fmeasure
            max_rouge = max(max_rouge, score)
        return round(max_rouge * 5.0, 2)  # Normalize to 0-5 scale

    def tool_selection_accuracy(self, pred_tool: str, gold_tool: str) -> bool:
        """Check if the predicted tool matches the expected tool."""
        return pred_tool == gold_tool

    def tool_call_success_rate(self, traces: List[Dict]) -> float:
        """Fraction of tool calls that executed without errors."""
        if not traces:
            return 1.0
        success = sum(1 for t in traces if "error" not in t.get("output", {}))
        return success / len(traces)

    def escalation_metrics(self, pred_escalate: bool, gold_escalate: bool) -> tuple:
        """Returns (TP, FP, FN) for escalation classification."""
        tp = 1 if pred_escalate and gold_escalate else 0
        fp = 1 if pred_escalate and not gold_escalate else 0
        fn = 1 if not pred_escalate and gold_escalate else 0
        return tp, fp, fn

    def text_metrics(self, preds: List[str], refs: List[str]) -> tuple:
        """Compute ROUGE-L between predictions and references."""
        if not preds or not refs:
            return 0.0, 0.0
        r_scores = [self.rouge.score(r, p)['rougeL'].fmeasure for p, r in zip(preds, refs)]
        avg_rouge = sum(r_scores) / len(r_scores) if r_scores else 0.0
        return avg_rouge, 0.0  # BERTScore omitted for speed; can be enabled with evaluate lib

    def mission_criticality_accuracy(self, pred_success_rate: float, true_success_rate: float) -> float:
        """
        Novel Metric: Mission-Criticality Accuracy (MCA).
        Returns 1.0 if the predicted success rate is within ±5% of the ground truth.
        This metric is specifically designed for aerospace applications where
        statistical accuracy directly impacts safety-critical decisions.
        
        MCA(p, g) = 1.0 if |p - g| <= 0.05, else 0.0
        """
        if abs(pred_success_rate - true_success_rate) <= 0.05:
            return 1.0
        return 0.0

    def compute_all(self, answer: str, passages: List[str], pred_tool: str,
                    gold_tool: str, pred_escalate: bool, gold_escalate: bool,
                    expected_keywords: List[str] = None) -> Dict[str, float]:
        """Compute all metrics in one call and return a dict."""
        results = {
            "faithfulness": self.faithfulness_score(answer, passages),
            "answer_in_kb": self.answer_in_kb_rate(answer, passages),
            "tool_correct": 1.0 if self.tool_selection_accuracy(pred_tool, gold_tool) else 0.0,
        }
        if expected_keywords:
            results["keyword_grounding"] = self.keyword_grounding_score(answer, expected_keywords)
        
        tp, fp, fn = self.escalation_metrics(pred_escalate, gold_escalate)
        results["escalation_tp"] = tp
        results["escalation_fp"] = fp
        results["escalation_fn"] = fn
        
        return results
