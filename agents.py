import json
from crewai import Agent, Task, Crew, Process, LLM
from tools import (
    fetch_weather_data,
    calculate_expected_pv_power,
    get_power_baseline
)

def run_multi_agent_pipeline(prediction: dict, site_id: str = "SITE_01", lat: float = 8.75, lon: float = 80.50, llm: LLM = None) -> str:

    # ----------------------------------------------------------------------
    # AGENT: SOLAR DIAGNOSTIC EXPERT
    # (single agent for now — validates the ML prediction against live
    #  weather + physics + SCADA history, and writes up the result as a
    #  proper technical ticket. tech_expert / financial_planner agents can
    #  be re-added later as extra Crew members + tasks with context=[task_ticket])
    # ----------------------------------------------------------------------
    solar_expert = Agent(
        role="Senior PV Diagnostic Expert",
        goal="Validate ML fault predictions against live weather, theoretical physics, and SCADA baselines, then issue a clear technical diagnostic ticket.",
        backstory="Senior Solar Reliability Engineer with 15+ years diagnosing tropical PV arrays, expert in irradiance physics, monsoon cloud derating, and meteorological verification of automated fault alerts.",
        tools=[fetch_weather_data, calculate_expected_pv_power, get_power_baseline],
        llm=llm,
        verbose=True
    )

    task_ticket = Task(
        description=f"""
        Site '{site_id}' at coordinates (Lat: {lat}, Lon: {lon}) has an ML fault
        detection of: {json.dumps(prediction)}

        Do the following, in order:
        1. Fetch current weather for the site coordinates.
        2. Calculate theoretical expected PV power output using 'Calculate Expected PV Power Output'.
        3. Query the historical SCADA baseline for this site.
        4. Compare the ML prediction against the weather-adjusted theoretical power and the historical baseline.
        5. Decide a status: CONFIRMED_FAULT, WEATHER_FALSE_POSITIVE, or LOW_CONFIDENCE, with a confidence score (0.0-1.0).
        6. Write up the finding as a formal technical ticket, in the voice of a
           senior solar reliability engineer, using EXACTLY this structure:

        **PV Diagnostic Ticket**
        - Site: {site_id}
        - Location: (Lat: {lat}, Lon: {lon})
        - ML Model Prediction: [fault type + model confidence]

        **Weather & Environmental Conditions**
        - Temperature: [°C]
        - Cloud Cover: [%]
        - Irradiance: [W/m²]

        **Physics & SCADA Validation**
        - Theoretical Expected Power: [W]
        - Historical Baseline (Median): [kW]
        - Deviation / Comparison: [1-2 sentences]

        **Diagnosis**
        - Status: [CONFIRMED_FAULT | WEATHER_FALSE_POSITIVE | LOW_CONFIDENCE]
        - Confidence: [0.0-1.0]
        - Justification: [2-3 sentences explaining the reasoning]

        **Recommended Next Step**
        - [One clear sentence: e.g. dispatch a technician for physical inspection,
          or no action needed and continue monitoring]
        """,
        expected_output="A fully filled-in technical ticket following the exact structure above, no placeholder brackets left unresolved.",
        agent=solar_expert
    )

    # ----------------------------------------------------------------------
    # CREW
    # ----------------------------------------------------------------------
    crew = Crew(
        agents=[solar_expert],
        tasks=[task_ticket],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()
    return str(result)
