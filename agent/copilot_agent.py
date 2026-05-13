import re
import json
import urllib.parse
from typing import List, Dict, Any
from tools.tool_loop import ToolExecutionLoop
import yaml

SYSTEM_PROMPT = """
You are SpaceCopilot, an expert AI assistant for space scientists and mission analysts.
1. CITATION-FIRST: Every factual claim must cite [doc_id] from retrieved passages.
2. HELPFUL: Answer the full question. Be precise and technical.
3. ESCALATE: If the KB has insufficient evidence, create a support ticket.
"""

# ── Relevance threshold: ChromaDB L2 distance above this = irrelevant ──
_KB_RELEVANCE_THRESHOLD = 1.2  # L2 distance; lower = more similar

# ── Space-domain keywords for out-of-scope guard ──
_SPACE_DOMAIN_KEYWORDS = [
    "space", "rocket", "satellite", "orbit", "launch", "nasa", "esa", "isro",
    "spacex", "falcon", "starship", "soyuz", "ariane", "atlas", "delta",
    "mars", "moon", "lunar", "jupiter", "venus", "mercury", "asteroid",
    "comet", "telescope", "hubble", "jwst", "james webb", "iss", "station",
    "payload", "leo", "geo", "gto", "meo", "trajectory", "mission",
    "propulsion", "engine", "thruster", "fuel", "booster", "capsule",
    "crew", "astronaut", "cosmonaut", "debris", "deorbit", "reentry",
    "starlink", "constellation", "communication", "sensor", "telemetry",
    "attitude", "control", "solar", "panel", "antenna", "rover",
    "perseverance", "curiosity", "ingenuity", "exploration", "deep space",
    "radiation", "magnetosphere", "exoplanet", "kepler", "transit",
    "spectroscopy", "infrared", "copernicus", "sentinel", "gaia",
    "gravity", "orbital mechanics", "delta-v", "hohmann", "transfer",
    "docking", "rendezvous", "eva", "spacewalk", "heat shield",
    "aerodynamics", "avionics", "flight", "altitude", "apogee", "perigee",
    "black hole", "neutron star", "galaxy", "nebula", "supernova",
]

# ── Explicit rate/probability keywords (checked FIRST) ──
_RATE_KEYWORDS = [
    "success rate", "failure rate", "failure probability",
    "probability of failure", "probability of success",
    "chance of failure", "chance of success",
    "how reliable", "reliability of", "calculate the",
    "what is the success", "what is the failure", "what is the chance",
    "historical failure", "historical success",
]

# ── Launch-window phrases ──
_WINDOW_KEYWORDS = [
    "launch window", "best time to launch", "when to launch",
    "optimal window", "next launch", "when can we launch",
]

# ── Policy phrases ──
_POLICY_KEYWORDS = [
    "policy", "regulation", "rule", "guideline",
    "standard", "compliance", "law",
]


def _is_space_domain(query: str) -> bool:
    """Check if the query is related to the space domain."""
    q = query.lower()
    # Very short queries (< 3 chars) are gibberish
    if len(q.strip()) < 3:
        return False
    return any(kw in q for kw in _SPACE_DOMAIN_KEYWORDS)


def _generate_google_url(query: str) -> str:
    """Generate a Google search URL scoped to space/NASA domain."""
    # Add space context to the search
    search_query = f"{query} site:nasa.gov OR site:esa.int OR site:spacex.com OR site:space.com"
    encoded = urllib.parse.quote_plus(search_query)
    return f"https://www.google.com/search?q={encoded}"


def _check_kb_relevance(tool_output: dict, query: str = "") -> bool:
    """Check if KB results are relevant based on distance scores.
    ChromaDB uses L2 distance — lower = more similar.
    Also checks keyword overlap between query and top results.
    Returns True if results are relevant, False if not."""
    passages = tool_output.get("passages", [])
    if not passages:
        return False
    
    # Check 1: L2 distance threshold
    top_score = passages[0].get("score", 999)
    if top_score > _KB_RELEVANCE_THRESHOLD:
        return False
    
    # Check 2: Keyword overlap — do the results actually mention query terms?
    # Extract meaningful words from the query (>= 3 chars, not stop words)
    _STOP_WORDS = {"the", "what", "how", "who", "when", "where", "why", "does",
                   "tell", "about", "give", "show", "find", "get", "can", "you",
                   "for", "and", "are", "was", "has", "have", "been", "from",
                   "with", "this", "that", "they", "will", "its", "any", "all",
                   "not", "but", "also", "into", "there", "which", "their", "some",
                   "more", "than", "our", "your", "his", "her", "she", "him"}
    
    query_words = set()
    for w in re.sub(r'[^a-z0-9\s]', '', query.lower()).split():
        if len(w) >= 3 and w not in _STOP_WORDS:
            query_words.add(w)
    
    if not query_words:
        return top_score < 0.8  # very short query — rely purely on distance
    
    # Check top 3 passages for keyword overlap
    for p in passages[:3]:
        text_lower = p.get("text", "").lower() + " " + p.get("title", "").lower()
        overlap = sum(1 for w in query_words if w in text_lower)
        if overlap >= 1:
            return True  # At least one meaningful keyword found in a passage
    
    # No keyword overlap found — results are noise
    return False


def _route_tool(query: str) -> tuple:
    """Returns (tool_name, tool_input_dict) based on keyword heuristics."""
    q = query.lower()

    # ── 1. Ticket escalation — HIGHEST priority ──
    if any(k in q for k in ["anomaly", "not working", "critical",
                             "issue", "problem", "report", "broken",
                             "stopped working", "failed to respond"]):
        return "CreateTicket", {
            "summary": query,
            "category": "Mission Anomaly",
            "severity": "High",
            "mission_context": query
        }

    # ── 2. Explicit success/failure rate queries ──
    if any(k in q for k in _RATE_KEYWORDS):
        vehicle = "Unknown"
        for name in ["falcon 9", "falcon heavy", "starship", "atlas v", "vulcan",
                     "delta iv", "soyuz", "ariane 5", "ariane 6", "ariane"]:
            if name in q:
                vehicle = name.title()
                break
        return "ComputeSuccessRate", {
            "mission_type": "orbital",
            "agency": "SpaceX" if "falcon" in q or "starship" in q else "NASA",
            "launch_vehicle": vehicle,
            "orbit": "LEO"
        }

    # ── 3. Launch window queries ──
    if any(k in q for k in _WINDOW_KEYWORDS):
        dest = "mars"
        for d in ["mars", "moon", "lunar", "jupiter", "venus", "iss", "leo", "geo"]:
            if d in q:
                dest = d
                break
        dest_label = dest.upper() if len(dest) <= 3 else dest.title()
        return "GetLaunchWindow", {
            "destination": dest_label,
            "spacecraft_mass_kg": 500.0,
            "launch_site": "KSC LC-39A",
            "target_date": "2026-01-01"
        }

    # ── 4. Policy queries ──
    if any(k in q for k in _POLICY_KEYWORDS):
        agency = "NASA"
        if "esa" in q:
            agency = "ESA"
        elif "isro" in q:
            agency = "ISRO"
        section = "debris" if "debris" in q else \
                  "safety"  if "safety" in q else \
                  "reuse"   if "reuse" in q or "reusable" in q else "general"
        return "GetPolicy", {"section_id": section, "agency": agency}

    # ── 5. Default: KB search ──
    return "SearchKB", {"query": query, "top_k": 5}


# ──────────────────────────────────────────────
# Answer formatter: converts raw tool output to a clean Markdown answer
# ──────────────────────────────────────────────
def _format_answer(tool_name: str, tool_input: dict, tool_output: dict, query: str) -> str:
    """Produces a rich Markdown answer from a tool result."""

    if tool_name == "SearchKB":
        passages = tool_output.get("passages", [])
        if not passages:
            return "⚠️ No relevant passages found in the knowledge base for your query."
        answer = "### 📡 Retrieved from Knowledge Base\n\n"
        for p in passages[:3]:
            doc_id = p.get("doc_id", "unknown")
            title  = p.get("title", "Untitled")
            text   = p.get("text", "")[:300]
            score  = p.get("score", 0.0)
            answer += f"**[{doc_id}]** _{title}_\n\n> {text}...\n\n"
        answer += f"\n*{len(passages)} passages retrieved from the offline space mission KB.*"
        return answer

    elif tool_name == "ComputeSuccessRate":
        sr   = tool_output.get("success_rate", 0.0)
        tot  = tool_output.get("total_missions", 0)
        modes = tool_output.get("failure_modes", [])
        conf = tool_output.get("confidence", 0.0)
        vehicle = tool_input.get("launch_vehicle", "selected vehicle")
        answer  = f"### 📊 Mission Success Analysis — {vehicle}\n\n"
        answer += f"| Metric | Value |\n|---|---|\n"
        answer += f"| **Success Rate** | **{sr*100:.1f}%** |\n"
        answer += f"| **Total Missions Analyzed** | {tot} |\n"
        answer += f"| **Confidence** | {conf*100:.0f}% |\n\n"
        if modes:
            answer += f"**Known Failure Modes:**\n"
            for m in modes:
                answer += f"- {m}\n"
        return answer

    elif tool_name == "GetLaunchWindow":
        windows = tool_output.get("optimal_windows", [])
        site    = tool_output.get("recommended_site", "N/A")
        risks   = tool_output.get("risk_factors", [])
        dv      = tool_output.get("delta_v_estimate", 0.0)
        dest    = tool_input.get("destination", "target")
        answer  = f"### 🚀 Launch Window Analysis — {dest}\n\n"
        answer += f"**Optimal Launch Windows:**\n"
        for w in windows:
            answer += f"- 🗓️ `{w}`\n"
        answer += f"\n**Recommended Launch Site:** {site}\n"
        answer += f"\n**Estimated Δv Required:** `{dv:,.0f} m/s`\n"
        if risks:
            answer += f"\n**Risk Factors:**\n"
            for r in risks:
                answer += f"- ⚠️ {r}\n"
        return answer

    elif tool_name == "GetPolicy":
        agency  = tool_output.get("agency", "Unknown Agency")
        section = tool_output.get("section_id", "N/A")
        text    = tool_output.get("policy_text", "No policy text available.")
        updated = tool_output.get("last_updated", "Unknown")
        answer  = f"### 📋 {agency} Policy — Section `{section}`\n\n"
        answer += f"> {text}\n\n"
        answer += f"*Last updated: {updated}*"
        return answer

    elif tool_name == "CreateTicket":
        ticket_id = tool_output.get("ticket_id", "N/A")
        eta       = tool_output.get("eta_hours", 72)
        answer    = f"### ⚠️ Support Ticket Escalated\n\n"
        answer   += f"Your issue has been escalated to a human mission analyst, as the knowledge base does not contain sufficient evidence to resolve this.\n\n"
        answer   += f"| Field | Value |\n|---|---|\n"
        answer   += f"| **Ticket ID** | `{ticket_id}` |\n"
        answer   += f"| **Status** | 🟡 Open |\n"
        answer   += f"| **ETA** | {eta} hours |\n"
        return answer

    return tool_output.get("answer", str(tool_output))


def _format_not_found(query: str, google_url: str, ticket_output: dict = None) -> str:
    """Format a 'not found in KB' response with Google search link and escalation ticket."""
    answer  = "### 🔍 Topic Not Found in Knowledge Base\n\n"
    answer += f"The query **\"{query}\"** did not match any relevant documents in our offline space mission knowledge base.\n\n"
    answer += "---\n\n"
    answer += "#### 🌐 External Search Suggested\n\n"
    answer += f"You can search for this topic on verified space agency sources:\n\n"
    answer += f"🔗 **[Search on Google (NASA / ESA / SpaceX)]({google_url})**\n\n"
    answer += "---\n\n"
    
    if ticket_output:
        ticket_id = ticket_output.get("ticket_id", "N/A")
        eta = ticket_output.get("eta_hours", 72)
        answer += "#### 🎫 Escalation Ticket Created\n\n"
        answer += f"A support ticket has been created for a subject-matter expert to review this query.\n\n"
        answer += f"| Field | Value |\n|---|---|\n"
        answer += f"| **Ticket ID** | `{ticket_id}` |\n"
        answer += f"| **Status** | 🟡 Open |\n"
        answer += f"| **ETA** | {eta} hours |\n"

    return answer


def _format_out_of_scope(query: str) -> str:
    """Format response for queries that are clearly not space-domain related."""
    answer  = "### 🚫 Out of Scope\n\n"
    answer += f"The query **\"{query}\"** does not appear to be related to the space domain.\n\n"
    answer += "I am **SpaceCopilot** — a specialized AI assistant for:\n\n"
    answer += "- 🚀 **Launch vehicles** — Falcon 9, Soyuz, Ariane, Starship, Atlas V\n"
    answer += "- 🛰️ **Satellites & missions** — ISS, Hubble, JWST, Starlink, Sentinel\n"
    answer += "- 📊 **Mission analytics** — Success rates, failure modes, reliability\n"
    answer += "- 🪐 **Orbital mechanics** — Launch windows, Δv, transfer orbits\n"
    answer += "- 📋 **Agency policies** — NASA, ESA, ISRO regulations & standards\n"
    answer += "- ⚠️ **Anomaly reporting** — Ticket escalation for mission-critical issues\n\n"
    answer += "*Please rephrase your query in the space domain context.*"
    return answer


class SpaceCopilotAgent:
    def __init__(self):
        with open("config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
        self.tool_loop = ToolExecutionLoop(
            max_iterations=self.config["tools"]["max_iterations"]
        )

    def process_query(self, query: str) -> Dict[str, Any]:
        """Main entry point for serving."""

        # ── Guard 1: Out-of-scope (gibberish / non-space queries) ──
        if not _is_space_domain(query):
            return {
                "tool_calls": [],
                "answer":     _format_out_of_scope(query),
                "citations":  [],
                "confidence": 0.0,
                "escalated":  False,
                "ticket_id":  None,
                "google_url": None
            }

        # Step 1: determine tool
        tool_name, tool_input = _route_tool(query)

        # Step 2: execute tool
        tool_output = self.tool_loop.registry.execute(tool_name, tool_input)

        # ── Guard 2: KB relevance check ──
        # If SearchKB returned results but they're all irrelevant (high L2 distance),
        # don't show them. Instead, generate Google URL + CreateTicket.
        if tool_name == "SearchKB" and not _check_kb_relevance(tool_output, query):
            google_url = _generate_google_url(query)
            
            # Auto-escalate: create a ticket for the missing knowledge
            ticket_input = {
                "summary": f"KB gap: {query}",
                "category": "Knowledge Gap",
                "severity": "Medium",
                "mission_context": query
            }
            ticket_output = self.tool_loop.registry.execute("CreateTicket", ticket_input)

            return {
                "tool_calls": [
                    {"tool": "SearchKB", "input": tool_input, "output": {"result": "No relevant matches"}},
                    {"tool": "CreateTicket", "input": ticket_input, "output": ticket_output},
                ],
                "answer":     _format_not_found(query, google_url, ticket_output),
                "citations":  [],
                "confidence": 0.0,
                "escalated":  True,
                "ticket_id":  ticket_output.get("ticket_id"),
                "google_url": google_url
            }

        # Step 3: format answer (normal path — relevant results found)
        answer = _format_answer(tool_name, tool_input, tool_output, query)

        # Step 4: extract citations only for KB searches
        citations = []
        if tool_name == "SearchKB":
            for p in tool_output.get("passages", [])[:3]:
                citations.append({
                    "doc_id": p.get("doc_id", ""),
                    "passage": p.get("text", "")[:150]
                })

        escalated  = tool_name == "CreateTicket"
        ticket_id  = tool_output.get("ticket_id") if escalated else None
        confidence = 0.95 if tool_name != "SearchKB" else 0.85

        return {
            "tool_calls": [{"tool": tool_name, "input": tool_input, "output": tool_output}],
            "answer":     answer,
            "citations":  citations,
            "confidence": confidence,
            "escalated":  escalated,
            "ticket_id":  ticket_id,
            "google_url": None
        }
