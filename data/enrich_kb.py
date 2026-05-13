"""
enrich_kb.py — Adds high-quality curated space mission documents to the ChromaDB
knowledge base so SearchKB returns useful, mission-specific results.
Run once: python -m data.enrich_kb
"""

import os
import uuid
import yaml
import chromadb

MISSIONS = [
    {
        "id": "FALCON9_HISTORY",
        "title": "SpaceX Falcon 9 Launch History",
        "text": (
            "The SpaceX Falcon 9 is a two-stage, partially reusable orbital rocket. "
            "First launched in June 2010, Falcon 9 has completed over 220 successful missions "
            "as of 2024, achieving a success rate above 98%. The rocket uses 9 Merlin engines "
            "on the first stage and 1 Merlin Vacuum engine on the second stage, burning RP-1 "
            "and liquid oxygen. Notable milestones include the first propulsive booster landing "
            "in December 2015, and reuse of a booster for the first time in March 2017. "
            "The only major failures were CRS-7 (2015, strut failure in second stage helium tank) "
            "and AMOS-6 (2016, pre-launch pad explosion). Falcon 9 has launched Crew Dragon, "
            "Starlink satellites, GPS III, and commercial communications satellites."
        )
    },
    {
        "id": "FALCON_HEAVY",
        "title": "SpaceX Falcon Heavy Overview",
        "text": (
            "Falcon Heavy is the world's most powerful operational rocket, consisting of three "
            "Falcon 9 cores strapped together. It produces 5 million pounds of thrust at liftoff. "
            "First launched February 2018, carrying Elon Musk's Tesla Roadster to heliocentric orbit. "
            "As of 2024, it has completed 9 missions with 1 partial failure (side boosters lost). "
            "It can lift 63.8 metric tons to LEO and 26.7 metric tons to GTO. Notable missions: "
            "Arabsat-6A, STP-2 (USAF payload), USSF-44, USSF-52, Jupiter-3. "
            "Success rate: 93.3% for complete mission success."
        )
    },
    {
        "id": "ISS_OVERVIEW",
        "title": "International Space Station",
        "text": (
            "The International Space Station (ISS) is a modular space station in LEO at 408 km altitude "
            "and 51.6° inclination. Operated jointly by NASA, Roscosmos, ESA, JAXA, and CSA. "
            "Assembly began in 1998 and has been continuously inhabited since November 2000. "
            "The station has a mass of approximately 420,000 kg and contains 15 pressurized modules. "
            "It orbits Earth every 92 minutes at 7.66 km/s. The station is planned for deorbit "
            "around 2030, with NASA selecting SpaceX to develop a deorbit vehicle. "
            "Commercial crew access is provided by SpaceX Crew Dragon and Boeing Starliner."
        )
    },
    {
        "id": "HUBBLE_OVERVIEW",
        "title": "Hubble Space Telescope",
        "text": (
            "The Hubble Space Telescope (HST) is a NASA/ESA space telescope launched April 24, 1990 "
            "aboard Space Shuttle Discovery. It orbits Earth at 547 km altitude. HST uses a 2.4-meter "
            "primary mirror and instruments spanning UV, visible, and near-infrared wavelengths. "
            "Five servicing missions (SM1-SM4B) corrected its initial mirror flaw and upgraded instruments. "
            "Hubble has produced over 1.3 million scientific observations and contributed to "
            "determining the Hubble constant, detecting dark energy, and characterizing exoplanet atmospheres. "
            "As of 2024, Hubble remains operational in a reduced gyroscope mode after gyro failures."
        )
    },
    {
        "id": "JWST_OVERVIEW",
        "title": "James Webb Space Telescope",
        "text": (
            "The James Webb Space Telescope (JWST) is the largest and most powerful space telescope "
            "ever launched, operated by NASA, ESA, and CSA. Launched December 25, 2021 on an Ariane 5 "
            "rocket. JWST orbits the L2 Lagrange point 1.5 million km from Earth. "
            "Its 6.5-meter segmented gold-coated beryllium primary mirror operates at 50 Kelvin. "
            "JWST observes in infrared (0.6–28.5 μm) using four instruments: NIRCam, NIRSpec, MIRI, FGS/NIRISS. "
            "First science images released July 2022. Mission lifetime: 20+ years with fuel reserve. "
            "Key science goals: first light objects, galaxy formation, exoplanet atmospheres, solar system."
        )
    },
    {
        "id": "ESA_SENTINEL",
        "title": "ESA Copernicus Sentinel Satellites",
        "text": (
            "The Copernicus programme is the European Union's Earth observation programme, managed by ESA. "
            "The Sentinel satellite family provides systematic observation of land, ocean, and atmosphere. "
            "Sentinel-1 (SAR radar, all-weather): S1A launched 2014, S1B 2016. "
            "Sentinel-2 (optical multispectral): S2A launched 2015, S2B 2017. 10/20/60m resolution. "
            "Sentinel-3 (ocean/land): S3A and S3B operating since 2016/2018. "
            "Sentinel-5P: atmospheric monitoring with Tropomi instrument. "
            "Sentinel-6 Michael Freilich: sea level altimetry. All operated by ESA from ESOC, Darmstadt."
        )
    },
    {
        "id": "ESA_MISSIONS",
        "title": "Key ESA Space Missions",
        "text": (
            "The European Space Agency (ESA) operates numerous science and exploration missions. "
            "Current active missions include: XMM-Newton (X-ray multi-mirror telescope, 1999–), "
            "Mars Express (Mars orbiter, 2003–), Rosetta/Philae (comet lander, ended 2016), "
            "Gaia (stellar astrometry, 1.7 billion stars mapped, 2013–), "
            "Solar Orbiter (solar physics, 2020–), Euclid (dark energy telescope, 2023–), "
            "BepiColombo (Mercury mission, 2018, arrival 2025), "
            "ExoMars (Mars atmospheric trace gas orbiter, 2016–). "
            "Upcoming: PLATO (exoplanet transits), LISA (gravitational waves)."
        )
    },
    {
        "id": "NASA_DEBRIS_POLICY",
        "title": "NASA Orbital Debris Mitigation Policy",
        "text": (
            "NASA Technical Standard NASA-STD-8719.14B addresses orbital debris mitigation. "
            "Key requirements: (1) Minimize debris released during normal operations; "
            "(2) Minimize the probability of on-orbit breakups; "
            "(3) Control post-mission disposal — LEO spacecraft must meet the 25-year deorbit rule; "
            "(4) Prevent on-orbit collisions — conjunction assessment and avoidance maneuvers required. "
            "GEO spacecraft must be raised to a graveyard orbit at least 300 km above GEO (35,786 km). "
            "The 1-in-10,000 casualty expectation limit applies to all re-entering objects. "
            "Post-mission passivation (venting residual propellants) is mandatory to prevent explosions."
        )
    },
    {
        "id": "SOYUZ_HISTORY",
        "title": "Soyuz Launch Vehicle History and Reliability",
        "text": (
            "The Soyuz rocket family is the world's most flown launch vehicle, with over 1,900 launches "
            "since its first flight in 1966. Operated by Roscosmos. Multiple variants: Soyuz-FG, Soyuz-2.1a, "
            "Soyuz-2.1b. Overall mission success rate exceeds 97.9%. "
            "Notable failure: Soyuz MS-10 (October 2018) — launch abort due to booster separation anomaly; "
            "both crew members safely returned via escape system. "
            "Soyuz has transported crews to ISS and launched Galileo navigation satellites for ESA. "
            "The Soyuz 7K-OK spacecraft first flew in 1967; the modern Soyuz MS variant is used for "
            "crew transport to ISS with a 6-hour orbital rendezvous capability."
        )
    },
    {
        "id": "ARIANE5_HISTORY",
        "title": "Ariane 5 Launch Vehicle",
        "text": (
            "Ariane 5 is an ESA heavy-lift launch vehicle operated by ArianeGroup / Arianespace. "
            "First launch: June 4, 1996 (failure). Entered operational service 1998. "
            "117 launches with success rate 97.4%. Two notable failures: "
            "Flight 501 (1996) — software error caused loss of attitude control and self-destruction; "
            "Flight VA241 (2018) — orbits were incorrect due to upper stage fault; satellites recovered. "
            "Can lift up to 10.5 tonnes to GTO or 21 tonnes to LEO depending on configuration. "
            "Used to launch JWST, Rosetta, BepiColombo, and XMM-Newton. "
            "Retired in 2023, succeeded by Ariane 6 (first flight July 2024)."
        )
    },
    {
        "id": "LEO_ORBIT",
        "title": "Low Earth Orbit (LEO) Parameters",
        "text": (
            "Low Earth Orbit (LEO) spans altitudes of approximately 200–2,000 km above Earth's surface. "
            "LEO satellites have orbital periods of 88–127 minutes and orbital velocities around 7.8 km/s. "
            "LEO is used for International Space Station (408 km), Hubble Space Telescope (547 km), "
            "Starlink constellation (340–560 km), Earth observation satellites, and crewed missions. "
            "Re-entry from LEO can be achieved with Δv of approximately 100 m/s retrograde burn. "
            "Atomic oxygen erosion and increased radiation from the South Atlantic Anomaly are key concerns. "
            "Under the 25-year rule, LEO spacecraft must deorbit within 25 years post-mission."
        )
    },
    {
        "id": "GEO_ORBIT",
        "title": "Geostationary Orbit (GEO) Parameters",
        "text": (
            "Geostationary orbit (GEO) is at exactly 35,786 km altitude, with an orbital period equal "
            "to Earth's rotation (23 hours, 56 minutes, 4 seconds). GEO satellites appear stationary "
            "relative to the ground. Used for communications satellites (DirectTV, SES, Intelsat), "
            "weather satellites (GOES, Meteosat), and navigation (SBAS). "
            "Launch requires GTO (Geostationary Transfer Orbit) followed by apogee kick to circularize. "
            "GEO end-of-life disposal requires raising to a 'graveyard orbit' ~300 km above GEO. "
            "Communication latency for GEO is approximately 240 ms round-trip."
        )
    },
    {
        "id": "STARLINK",
        "title": "SpaceX Starlink Satellite Constellation",
        "text": (
            "Starlink is a satellite internet constellation operated by SpaceX, designed to provide "
            "global broadband internet coverage. Over 5,000 satellites deployed as of 2024, "
            "operating in LEO at altitudes between 340 km and 560 km. "
            "Each Starlink satellite has a mass of approximately 306 kg (v1.5) or 800 kg (v2 Mini). "
            "The constellation uses Ku-band and Ka-band frequencies with inter-satellite laser links. "
            "User terminal download speeds: 100–300 Mbps with latency of 20–40 ms. "
            "Starlink Phase-1 operates at 53° inclination for mid-latitude coverage. "
            "End-of-life deorbit is achieved via propulsion within 5 years, meeting NASA debris standards."
        )
    },
    {
        "id": "MARS_MISSION",
        "title": "Mars Exploration Missions Overview",
        "text": (
            "Mars has been the target of numerous robotic exploration missions. Successful missions include: "
            "NASA Mars Science Laboratory (Curiosity Rover, landed 2012, operational), "
            "NASA Perseverance Rover (landed February 2021, Jezero Crater, collecting samples), "
            "NASA Ingenuity Helicopter (first powered flight on another planet, April 2021), "
            "ESA Mars Express (orbital science, 2003–present), "
            "NASA MAVEN (atmospheric science, 2014–present), "
            "NASA InSight (interior seismometer, landed 2018, mission ended December 2022). "
            "Mars to Earth communication delay: 4–24 minutes depending on orbital geometry. "
            "Next Mars launch window: October 2026 (approximate). Transfer takes ~7 months."
        )
    },
    {
        "id": "PAYLOAD_CAPABILITIES",
        "title": "Launch Vehicle Payload Capabilities Comparison",
        "text": (
            "Payload capacity comparison for major operational launch vehicles (2024): "
            "SpaceX Falcon 9: 22,800 kg to LEO, 8,300 kg to GTO (expendable). "
            "SpaceX Falcon Heavy: 63,800 kg to LEO, 26,700 kg to GTO, 16,800 kg to Mars. "
            "SpaceX Starship (target): 100,000–150,000 kg to LEO (fully reusable). "
            "ULA Atlas V: 18,850 kg to LEO, 8,900 kg to GTO. "
            "ULA Vulcan Centaur: 27,200 kg to LEO, 18,000 kg to GTO. "
            "Ariane 6 (A64): 21,650 kg to LEO, 11,500 kg to GTO. "
            "ISRO LVM3: 8,000 kg to LEO, 4,000 kg to GTO. "
            "Roscosmos Soyuz-2.1b: 8,200 kg to LEO."
        )
    },
]


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    kb_dir = config["paths"]["kb_dir"]
    os.makedirs(kb_dir, exist_ok=True)

    client = chromadb.PersistentClient(path=kb_dir)

    # Get or create the collection
    try:
        col = client.get_collection("space_kb")
        print(f"Found existing KB with {col.count()} documents.")
    except Exception:
        col = client.create_collection("space_kb")
        print("Created new KB collection.")

    # Check what's already in there
    existing_ids = set()
    if col.count() > 0:
        existing = col.get()
        existing_ids = set(existing["ids"])

    added = 0
    for doc in MISSIONS:
        if doc["id"] in existing_ids:
            print(f"  ⏭  Skipping {doc['id']} (already in KB)")
            continue
        col.add(
            ids=[doc["id"]],
            documents=[doc["text"]],
            metadatas=[{"title": doc["title"], "source": "curated"}]
        )
        print(f"  ✅ Added: {doc['title']}")
        added += 1

    print(f"\nDone. Added {added} documents. Total KB size: {col.count()}")


if __name__ == "__main__":
    main()
