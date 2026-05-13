"""
benchmark.py — Latency and throughput benchmarking for the Space Copilot API.
Measures:
  - Per-request latency (avg, p50, p95, p99)
  - Throughput (queries/second)
  - Breakdown by tool type

Usage:
  python -m evaluation.benchmark           # Direct agent benchmark (no server needed)
  python -m evaluation.benchmark --api     # API benchmark (server must be running)
"""
import time
import json
import sys
import os
import statistics
import argparse

BENCHMARK_QUERIES = [
    # SearchKB
    {"query": "Tell me about SpaceX Falcon 9 launch history", "expected_tool": "SearchKB"},
    {"query": "What is the International Space Station?", "expected_tool": "SearchKB"},
    {"query": "Describe the JWST mission", "expected_tool": "SearchKB"},
    # ComputeSuccessRate
    {"query": "What is the success rate of Falcon 9?", "expected_tool": "ComputeSuccessRate"},
    {"query": "Calculate the reliability of Soyuz launches", "expected_tool": "ComputeSuccessRate"},
    {"query": "What is the success rate of Ariane 5?", "expected_tool": "ComputeSuccessRate"},
    # GetLaunchWindow
    {"query": "What is the next optimal launch window to Mars?", "expected_tool": "GetLaunchWindow"},
    {"query": "When should we launch to the Moon?", "expected_tool": "GetLaunchWindow"},
    # GetPolicy
    {"query": "Is there a NASA policy regarding orbital debris?", "expected_tool": "GetPolicy"},
    {"query": "What are ESA guidelines for end-of-life disposal?", "expected_tool": "GetPolicy"},
    # CreateTicket
    {"query": "Critical anomaly in attitude control system. Not working.", "expected_tool": "CreateTicket"},
    # Out-of-scope
    {"query": "what is maths?", "expected_tool": "OutOfScope"},
]


def benchmark_direct(num_runs: int = 3):
    """Benchmark by calling the agent directly (no HTTP overhead)."""
    from agent.copilot_agent import SpaceCopilotAgent

    print("=" * 60)
    print("  SpaceCopilot Latency & Throughput Benchmark")
    print("  Mode: Direct Agent (no HTTP overhead)")
    print(f"  Queries: {len(BENCHMARK_QUERIES)} x {num_runs} runs = {len(BENCHMARK_QUERIES) * num_runs} total")
    print("=" * 60)

    agent = SpaceCopilotAgent()

    # Warm-up run
    print("\n⏳ Warm-up run...")
    agent.process_query("Hello")

    all_latencies = []
    tool_latencies = {}

    for run in range(num_runs):
        print(f"\n── Run {run+1}/{num_runs} ──")
        for bq in BENCHMARK_QUERIES:
            start = time.perf_counter()
            result = agent.process_query(bq["query"])
            elapsed_ms = (time.perf_counter() - start) * 1000
            all_latencies.append(elapsed_ms)

            tool = bq["expected_tool"]
            if tool not in tool_latencies:
                tool_latencies[tool] = []
            tool_latencies[tool].append(elapsed_ms)

            print(f"  {bq['query'][:45]:45s} | {elapsed_ms:7.1f}ms | {tool}")

    # ── Aggregate Results ──
    all_latencies.sort()
    total_time = sum(all_latencies) / 1000  # seconds
    throughput = len(all_latencies) / total_time if total_time > 0 else 0

    print("\n" + "=" * 60)
    print("  BENCHMARK RESULTS")
    print("=" * 60)
    print(f"\n  Total Requests:    {len(all_latencies)}")
    print(f"  Total Time:        {total_time:.2f}s")
    print(f"  Throughput:        {throughput:.1f} queries/sec")
    print(f"\n  {'Metric':<20s} {'Value':>10s}")
    print(f"  {'-'*35}")
    print(f"  {'Avg Latency':<20s} {statistics.mean(all_latencies):>8.1f}ms")
    print(f"  {'Median (P50)':<20s} {statistics.median(all_latencies):>8.1f}ms")
    print(f"  {'P95 Latency':<20s} {all_latencies[int(0.95 * len(all_latencies))]:>8.1f}ms")
    print(f"  {'P99 Latency':<20s} {all_latencies[int(0.99 * len(all_latencies))]:>8.1f}ms")
    print(f"  {'Min':<20s} {min(all_latencies):>8.1f}ms")
    print(f"  {'Max':<20s} {max(all_latencies):>8.1f}ms")

    print(f"\n  {'Tool':<25s} {'Avg (ms)':>10s} {'Count':>8s}")
    print(f"  {'-'*45}")
    for tool, lats in sorted(tool_latencies.items()):
        print(f"  {tool:<25s} {statistics.mean(lats):>8.1f}ms {len(lats):>6d}")

    # Save results
    results = {
        "mode": "direct",
        "total_requests": len(all_latencies),
        "total_time_sec": round(total_time, 3),
        "throughput_qps": round(throughput, 2),
        "avg_latency_ms": round(statistics.mean(all_latencies), 2),
        "median_latency_ms": round(statistics.median(all_latencies), 2),
        "p95_latency_ms": round(all_latencies[int(0.95 * len(all_latencies))], 2),
        "p99_latency_ms": round(all_latencies[int(0.99 * len(all_latencies))], 2),
        "min_latency_ms": round(min(all_latencies), 2),
        "max_latency_ms": round(max(all_latencies), 2),
        "per_tool": {
            tool: {
                "avg_ms": round(statistics.mean(lats), 2),
                "count": len(lats)
            } for tool, lats in tool_latencies.items()
        }
    }
    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✅ Results saved to evaluation/benchmark_results.json")
    print("=" * 60)


def benchmark_api(num_runs: int = 3, base_url: str = "http://localhost:8000"):
    """Benchmark via HTTP API (server must be running)."""
    import requests

    print("=" * 60)
    print("  SpaceCopilot API Benchmark")
    print(f"  Mode: HTTP API ({base_url})")
    print(f"  Queries: {len(BENCHMARK_QUERIES)} x {num_runs} runs")
    print("=" * 60)

    # Health check
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        print(f"  Server status: {r.json().get('status', 'unknown')}")
    except Exception as e:
        print(f"  ❌ Server not reachable: {e}")
        print(f"  Start the server first: python -m uvicorn serving.serve:app --port 8000")
        return

    all_latencies = []
    for run in range(num_runs):
        print(f"\n── Run {run+1}/{num_runs} ──")
        for bq in BENCHMARK_QUERIES:
            start = time.perf_counter()
            try:
                r = requests.post(f"{base_url}/query",
                                  json={"question": bq["query"]}, timeout=30)
                r.raise_for_status()
            except Exception as e:
                print(f"  ❌ {bq['query'][:40]:40s} | ERROR: {e}")
                continue
            elapsed_ms = (time.perf_counter() - start) * 1000
            all_latencies.append(elapsed_ms)
            server_latency = r.json().get("latency_ms", 0)
            print(f"  {bq['query'][:40]:40s} | Total: {elapsed_ms:7.1f}ms | Server: {server_latency:6.1f}ms")

    if all_latencies:
        all_latencies.sort()
        total_time = sum(all_latencies) / 1000
        print(f"\n  Avg Latency:  {statistics.mean(all_latencies):.1f}ms")
        print(f"  P95 Latency:  {all_latencies[int(0.95 * len(all_latencies))]:.1f}ms")
        print(f"  Throughput:   {len(all_latencies) / total_time:.1f} queries/sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpaceCopilot Benchmark")
    parser.add_argument("--api", action="store_true", help="Benchmark via HTTP API")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs")
    args = parser.parse_args()

    if args.api:
        benchmark_api(num_runs=args.runs)
    else:
        benchmark_direct(num_runs=args.runs)
