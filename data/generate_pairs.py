"""
generate_pairs.py — Generates realistic, diverse training data for ALL model components:
  1. DPO preference pairs  (dpo_pairs.jsonl)
  2. Tool policy labels     (tool_policy.jsonl)
  3. Retriever pairs        (retriever_pairs.jsonl)
  4. Reranker pairs         (reranker_pairs.jsonl)
  5. Generator SFT triples  (generator_sft.jsonl)
"""
import json
import os
import yaml
import random
import itertools

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

# ── Rich domain knowledge for data generation ──
VEHICLES = ["Falcon 9", "Falcon Heavy", "Starship", "Soyuz", "Ariane 5",
            "Atlas V", "Delta IV", "Vulcan Centaur", "PSLV", "LVM3"]

MISSIONS = [
    {"name": "Apollo 11", "agency": "NASA", "year": 1969, "desc": "First crewed lunar landing mission. Neil Armstrong and Buzz Aldrin walked on the Moon on July 20, 1969."},
    {"name": "Voyager 1", "agency": "NASA", "year": 1977, "desc": "Launched in 1977, Voyager 1 is the farthest human-made object from Earth, currently in interstellar space."},
    {"name": "Hubble Space Telescope", "agency": "NASA/ESA", "year": 1990, "desc": "Space telescope launched in 1990 with a 2.4m primary mirror. Five servicing missions corrected its initial mirror flaw."},
    {"name": "Mars Perseverance Rover", "agency": "NASA", "year": 2021, "desc": "Landed in Jezero Crater on Mars in February 2021. Equipped with MOXIE oxygen generator and Ingenuity helicopter."},
    {"name": "James Webb Space Telescope", "agency": "NASA/ESA/CSA", "year": 2021, "desc": "Largest space telescope ever built. 6.5m segmented mirror. Operates at L2 Lagrange point observing in infrared."},
    {"name": "Chandrayaan-3", "agency": "ISRO", "year": 2023, "desc": "India's third lunar mission. Successfully soft-landed near the south pole of the Moon on August 23, 2023."},
    {"name": "Cassini-Huygens", "agency": "NASA/ESA", "year": 1997, "desc": "Orbited Saturn from 2004 to 2017. The Huygens probe landed on Titan. Discovered geysers on Enceladus."},
    {"name": "Rosetta", "agency": "ESA", "year": 2004, "desc": "First mission to orbit and land on a comet (67P/Churyumov-Gerasimenko). Philae lander touched down in 2014."},
    {"name": "International Space Station", "agency": "Multi-agency", "year": 1998, "desc": "Modular space station in LEO at 408 km altitude. Continuously inhabited since November 2000. Mass: 420,000 kg."},
    {"name": "Starlink", "agency": "SpaceX", "year": 2019, "desc": "Satellite internet constellation with over 5,000 satellites in LEO. Provides global broadband coverage."},
    {"name": "Gaia", "agency": "ESA", "year": 2013, "desc": "Space observatory mapping 1.7 billion stars in the Milky Way with unprecedented precision in astrometry."},
    {"name": "New Horizons", "agency": "NASA", "year": 2006, "desc": "First spacecraft to fly by Pluto in July 2015. Discovered Pluto's heart-shaped nitrogen ice plain (Sputnik Planitia)."},
]

POLICIES = [
    {"agency": "NASA", "topic": "orbital debris", "text": "NASA-STD-8719.14B requires LEO spacecraft to deorbit within 25 years of end-of-life."},
    {"agency": "NASA", "topic": "crew safety", "text": "NPR 8715.3 mandates probability of loss of crew below 1 in 270 for crewed missions."},
    {"agency": "NASA", "topic": "launch vehicle reuse", "text": "NPD 8610.7F requires flight readiness review and vehicle health certification between reuse flights."},
    {"agency": "ESA", "topic": "space debris", "text": "ESA ESSB-ST-U-007 requires passivation of all stored energy sources at end-of-life."},
    {"agency": "ESA", "topic": "crew safety", "text": "ESA REG/003 classifies safety-critical software under SIL-4 integrity level."},
    {"agency": "ISRO", "topic": "debris mitigation", "text": "ISRO SDMP-01 follows IADC guidelines with 25-year LEO post-mission disposal requirement."},
]

DESTINATIONS = ["Mars", "Moon", "Jupiter", "Venus", "ISS", "LEO", "GEO", "Lunar orbit"]

ANOMALY_TEMPLATES = [
    "My satellite's {sys} is showing {fault}. Need immediate support.",
    "We detected a critical {fault} in the {sys} during {phase}.",
    "The {sys} on our spacecraft has {fault}. This is mission-critical.",
    "Reporting an anomaly: {sys} experienced {fault} at T+{t} seconds.",
    "Emergency: {sys} failure detected. {fault}. Requesting escalation.",
]
SYSTEMS = ["attitude control system", "thermal management unit", "solar array deployment",
           "communication subsystem", "propulsion module", "power distribution unit",
           "reaction wheel assembly", "star tracker", "gyroscope", "battery management system"]
FAULTS = ["intermittent signal loss", "unexpected torque readings", "temperature exceedance",
          "voltage drop below threshold", "telemetry data corruption", "sensor drift beyond limits",
          "mechanical vibration anomaly", "software watchdog timeout", "memory parity error"]
PHASES = ["orbital insertion", "commissioning", "science operations", "deorbit burn",
          "docking approach", "eclipse transition", "momentum dumping"]

# ── Query templates for each tool ──
SEARCH_TEMPLATES = [
    "Tell me about {mission}", "What is {mission}?", "Describe the {mission} mission",
    "Give me details about {mission}", "What are the key achievements of {mission}?",
    "Explain the significance of {mission}", "What do we know about {mission}?",
    "History of {mission}", "Overview of the {mission} project",
    "What instruments does {mission} carry?", "When was {mission} launched?",
    "Who built {mission}?", "What agency operates {mission}?",
]

RATE_TEMPLATES = [
    "What is the success rate of {vehicle}?",
    "Calculate the reliability of {vehicle} launches",
    "What is the historical failure probability of {vehicle}?",
    "How reliable is the {vehicle} rocket?",
    "What is the chance of failure for {vehicle}?",
    "Show me {vehicle} mission success statistics",
    "What is the probability of success for a {vehicle} launch?",
]

WINDOW_TEMPLATES = [
    "What is the next optimal launch window to {dest}?",
    "When should we launch a mission to {dest}?",
    "Best time to launch a {mass}kg payload to {dest}?",
    "Calculate the next launch window for {dest}",
    "When can we launch to {dest}?",
    "What are the upcoming {dest} transfer windows?",
]

POLICY_TEMPLATES = [
    "What is {agency}'s policy on {topic}?",
    "Are there {agency} regulations for {topic}?",
    "What guidelines does {agency} have regarding {topic}?",
    "Show me the {agency} standard for {topic}",
    "Is there a {agency} compliance rule for {topic}?",
    "What are the {agency} rules about {topic}?",
]

GREETING_TEMPLATES = [
    "Hello SpaceCopilot", "Hi there!", "Good morning", "Hey, how are you?",
    "What can you do?", "Hello", "Thanks!", "Goodbye",
]


# ──────────────────────────────────────────────
# 1. DPO Preference Pairs
# ──────────────────────────────────────────────
def generate_dpo_pairs(out_file: str, num_pairs: int = 150):
    pairs = []
    
    # Category A: Citation-grounded answers (chosen) vs hallucinated (rejected)
    for m in MISSIONS:
        query = random.choice(SEARCH_TEMPLATES).format(mission=m["name"])
        chosen = f"According to mission records [{m['name'].replace(' ', '_').upper()}_001], {m['desc']} The mission was operated by {m['agency']} starting in {m['year']}."
        rejected = f"{m['name']} is a space thing. It went to space and did science stuff. It was pretty cool."
        pairs.append({"query": query, "chosen": chosen, "rejected": rejected})
        
        # Variant: concise vs verbose
        query2 = f"When was {m['name']} launched?"
        chosen2 = f"{m['name']} was launched in {m['year']} by {m['agency']} [{m['name'].replace(' ', '_').upper()}_001]."
        rejected2 = f"Well, you see, {m['name']} is a really interesting mission. There's a lot to talk about. Space exploration has been going on for decades. Let me tell you a long story about it. Eventually, {m['name']} was launched sometime in the past."
        pairs.append({"query": query2, "chosen": chosen2, "rejected": rejected2})
    
    # Category B: Tool-routed answers (chosen) vs tool-less answers (rejected)
    for v in VEHICLES:
        query = f"What is the success rate of {v}?"
        chosen = f"Based on mission analytics [STATS_{v.replace(' ', '_').upper()}], {v} has demonstrated a strong track record. The computed success rate accounts for all documented launches including any partial failures."
        rejected = f"I think {v} has like a 90-something percent success rate? I'm not sure of the exact number but it's pretty reliable I guess."
        pairs.append({"query": query, "chosen": chosen, "rejected": rejected})
    
    # Category C: Proper escalation (chosen) vs making stuff up (rejected)
    for sys in SYSTEMS[:6]:
        query = f"My {sys} is malfunctioning. What should I do?"
        chosen = f"I've created an escalation ticket for your {sys} issue. A human mission analyst will review this within 24 hours. Ticket ID: TICKET-XXXXXX. In the meantime, please document any additional telemetry data."
        rejected = f"Oh don't worry about the {sys}, it's probably just a minor glitch. Try turning it off and on again. Space hardware does that sometimes."
        pairs.append({"query": query, "chosen": chosen, "rejected": rejected})
    
    # Category D: Policy-cited answers (chosen) vs vague answers (rejected)
    for p in POLICIES:
        query = f"What is {p['agency']}'s policy on {p['topic']}?"
        chosen = f"According to {p['agency']} policy [{p['agency']}_{p['topic'].replace(' ', '_').upper()}_POLICY], {p['text']}"
        rejected = f"{p['agency']} probably has some rules about {p['topic']} but I'm not exactly sure what they are. Space agencies generally have strict guidelines."
        pairs.append({"query": query, "chosen": chosen, "rejected": rejected})

    # Category E: Launch window with data (chosen) vs guess (rejected)
    for dest in DESTINATIONS[:5]:
        query = f"When is the next launch window to {dest}?"
        chosen = f"Based on orbital mechanics calculations [WINDOW_{dest.upper()}], the next optimal transfer window to {dest} depends on the synodic period. The recommended launch site and delta-v estimates have been computed."
        rejected = f"You can probably launch to {dest} anytime, just point the rocket that way and go. The window is whenever you want."
        pairs.append({"query": query, "chosen": chosen, "rejected": rejected})
    
    # Fill remaining with variations
    while len(pairs) < num_pairs:
        m = random.choice(MISSIONS)
        v = random.choice(VEHICLES)
        query = random.choice([
            f"Compare {m['name']} with other {m['agency']} missions",
            f"What are the risks of launching {v} to {random.choice(DESTINATIONS)}?",
            f"Should I use {v} for a {random.randint(100,5000)}kg payload?",
        ])
        chosen = f"Based on the knowledge base evidence, {m['name']} operated by {m['agency']} since {m['year']} provides relevant context. {m['desc'][:100]}. [Source: KB_{m['name'].replace(' ', '_').upper()}]"
        rejected = f"I don't really know much about that. Maybe try Google?"
        pairs.append({"query": query, "chosen": chosen, "rejected": rejected})
    
    random.shuffle(pairs)
    with open(out_file, "w") as f:
        for p in pairs[:num_pairs]:
            f.write(json.dumps(p) + "\n")
    print(f"  ✅ Generated {min(len(pairs), num_pairs)} DPO pairs")


# ──────────────────────────────────────────────
# 2. Tool Policy Classification Data
# ──────────────────────────────────────────────
def generate_tool_policy_data(out_file: str, num_samples: int = 600):
    samples = []
    
    # SearchKB queries
    for m in MISSIONS:
        for tmpl in SEARCH_TEMPLATES:
            samples.append({"query": tmpl.format(mission=m["name"]), "label": "SearchKB"})
    
    # ComputeSuccessRate queries
    for v in VEHICLES:
        for tmpl in RATE_TEMPLATES:
            samples.append({"query": tmpl.format(vehicle=v), "label": "ComputeSuccessRate"})
    
    # GetLaunchWindow queries
    for dest in DESTINATIONS:
        for tmpl in WINDOW_TEMPLATES:
            mass = random.randint(100, 10000)
            samples.append({"query": tmpl.format(dest=dest, mass=mass), "label": "GetLaunchWindow"})
    
    # GetPolicy queries
    for p in POLICIES:
        for tmpl in POLICY_TEMPLATES:
            samples.append({"query": tmpl.format(agency=p["agency"], topic=p["topic"]), "label": "GetPolicy"})
    
    # CreateTicket queries
    for _ in range(80):
        sys = random.choice(SYSTEMS)
        fault = random.choice(FAULTS)
        phase = random.choice(PHASES)
        t = random.randint(10, 3600)
        tmpl = random.choice(ANOMALY_TEMPLATES)
        q = tmpl.format(sys=sys, fault=fault, phase=phase, t=t)
        samples.append({"query": q, "label": "CreateTicket"})
    
    # NoTool / greeting
    for g in GREETING_TEMPLATES:
        samples.append({"query": g, "label": "NoTool"})
    
    random.shuffle(samples)
    with open(out_file, "w") as f:
        for s in samples[:num_samples]:
            f.write(json.dumps(s) + "\n")
    print(f"  ✅ Generated {min(len(samples), num_samples)} tool policy samples")


# ──────────────────────────────────────────────
# 3. Retriever Contrastive Pairs
# ──────────────────────────────────────────────
def generate_retriever_pairs(out_file: str):
    pairs = []
    
    for m in MISSIONS:
        # Positive pair: query matches mission description
        queries = [
            f"What is {m['name']}?",
            f"Tell me about {m['name']}",
            f"When was {m['name']} launched?",
            f"{m['name']} mission details",
            f"Who operates {m['name']}?",
        ]
        for q in queries:
            pairs.append({"query": q, "positive": m["desc"], "source": m["name"]})
    
    # Cross-domain pairs: policy queries → policy text
    for p in POLICIES:
        queries = [
            f"What is {p['agency']}'s {p['topic']} policy?",
            f"{p['agency']} {p['topic']} regulation",
            f"Rules about {p['topic']} from {p['agency']}",
        ]
        for q in queries:
            pairs.append({"query": q, "positive": p["text"], "source": f"{p['agency']}_{p['topic']}"})
    
    # Vehicle queries
    vehicle_facts = {
        "Falcon 9": "The SpaceX Falcon 9 is a partially reusable two-stage rocket with over 220 successful missions and a 98.4% success rate.",
        "Falcon Heavy": "Falcon Heavy is the world's most powerful operational rocket with 5 million pounds of thrust, consisting of three Falcon 9 cores.",
        "Soyuz": "The Soyuz rocket family has over 1,900 launches since 1966 with a 97.9% mission success rate, the most flown launch vehicle ever.",
        "Ariane 5": "Ariane 5 is an ESA heavy-lift launcher with 117 launches and 97.4% success rate. It launched JWST, Rosetta, and BepiColombo.",
        "Starship": "SpaceX Starship is the largest and most powerful rocket ever built, designed for Mars colonization with 100+ tonne LEO capacity.",
    }
    for v, fact in vehicle_facts.items():
        queries = [f"What is {v}?", f"{v} specifications", f"Tell me about {v}", f"{v} launch history"]
        for q in queries:
            pairs.append({"query": q, "positive": fact, "source": v})
    
    with open(out_file, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"  ✅ Generated {len(pairs)} retriever training pairs")


# ──────────────────────────────────────────────
# 4. Reranker Pairs (positive + negative)
# ──────────────────────────────────────────────
def generate_reranker_pairs(out_file: str):
    pairs = []
    
    for i, m in enumerate(MISSIONS):
        query = f"Tell me about {m['name']}"
        # Positive
        pairs.append({"query": query, "passage": m["desc"], "label": 1})
        # Hard negative: different mission
        neg = MISSIONS[(i + 3) % len(MISSIONS)]
        pairs.append({"query": query, "passage": neg["desc"], "label": 0})
        # Easy negative: random unrelated text
        pairs.append({"query": query, "passage": "The Amazon rainforest covers 5.5 million square kilometers in South America.", "label": 0})
        pairs.append({"query": query, "passage": "Python is a popular programming language for data science and machine learning.", "label": 0})
    
    for p in POLICIES:
        query = f"{p['agency']} {p['topic']} policy"
        pairs.append({"query": query, "passage": p["text"], "label": 1})
        neg_p = random.choice([x for x in POLICIES if x != p])
        pairs.append({"query": query, "passage": neg_p["text"], "label": 0})
    
    random.shuffle(pairs)
    with open(out_file, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"  ✅ Generated {len(pairs)} reranker training pairs")


# ──────────────────────────────────────────────
# 5. Generator SFT Triples (query, context, answer)
# ──────────────────────────────────────────────
def generate_generator_sft(out_file: str):
    triples = []
    
    for m in MISSIONS:
        doc_id = m["name"].replace(" ", "_").upper()
        # Factual Q&A
        triples.append({
            "text": f"Query: What is {m['name']}? Context: [{doc_id}: {m['desc']}] Answer: {m['desc']} [{doc_id}]."
        })
        triples.append({
            "text": f"Query: When was {m['name']} launched? Context: [{doc_id}: {m['desc']}] Answer: {m['name']} was launched in {m['year']} by {m['agency']} [{doc_id}]."
        })
        triples.append({
            "text": f"Query: Which agency operates {m['name']}? Context: [{doc_id}: {m['desc']}] Answer: {m['name']} is operated by {m['agency']} [{doc_id}]."
        })
    
    for p in POLICIES:
        doc_id = f"{p['agency']}_{p['topic'].replace(' ', '_').upper()}"
        triples.append({
            "text": f"Query: What is {p['agency']}'s policy on {p['topic']}? Context: [{doc_id}: {p['text']}] Answer: {p['text']} [{doc_id}]."
        })
    
    # Escalation examples
    for sys in SYSTEMS[:5]:
        triples.append({
            "text": f"Query: My {sys} is not working. Context: [No relevant KB passages found.] Answer: I have escalated your issue to a human mission analyst. A support ticket has been created. [ESCALATED]"
        })
    
    with open(out_file, "w") as f:
        for t in triples:
            f.write(json.dumps(t) + "\n")
    print(f"  ✅ Generated {len(triples)} generator SFT triples")


# ──────────────────────────────────────────────
def main():
    config = load_config()
    data_dir = config["paths"]["data_dir"]
    os.makedirs(data_dir, exist_ok=True)
    
    print("=" * 55)
    print("  Generating Training Data for All Model Components")
    print("=" * 55)
    
    generate_dpo_pairs(os.path.join(data_dir, "dpo_pairs.jsonl"), num_pairs=150)
    generate_tool_policy_data(os.path.join(data_dir, "tool_policy.jsonl"), num_samples=600)
    generate_retriever_pairs(os.path.join(data_dir, "retriever_pairs.jsonl"))
    generate_reranker_pairs(os.path.join(data_dir, "reranker_pairs.jsonl"))
    generate_generator_sft(os.path.join(data_dir, "generator_sft.jsonl"))
    
    print("\n✅ All training data generated successfully!")
    print(f"   Data directory: {os.path.abspath(data_dir)}")

if __name__ == "__main__":
    main()
