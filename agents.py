import json
from crewai import Agent, Task, Crew, Process, LLM
from tools import get_coordinates_for_location, fetch_weather_data

def run_multi_agent_pipeline(prediction: dict, site_id: str = "SITE_01", location_name: str = "Kilinochchi", llm: LLM = None) -> str:

    # ----------------------------------------------------------------------
    # AGENT: SOLAR DIAGNOSTIC EXPERT
    # (single agent -- resolves the site's coordinates itself, checks live
    #  weather, and reasons about whether it plausibly explains the ML fault
    #  detection. No power-output math, no SCADA baseline: just location +
    #  weather + judgment.)
    # ----------------------------------------------------------------------
    solar_expert = Agent(
        role="Senior PV Diagnostic Expert",
        goal="Resolve the site's coordinates, check current weather there, and use it to sanity-check an ML fault detection, then issue a clear diagnostic ticket.",
        backstory="Senior Solar Reliability Engineer with 15+ years diagnosing tropical PV arrays, skilled at judging whether weather conditions like heavy cloud cover, rain, or low light could plausibly be behind an automated fault alert.",
        tools=[get_coordinates_for_location, fetch_weather_data],
        llm=llm,
        verbose=True
    )

    task_ticket = Task(
        description=f"""
        Site '{site_id}' is located in the '{location_name}' district of Sri Lanka.
        The ML fault detection is: {json.dumps(prediction)}

        Do the following, in order:
        1. Use 'Get Coordinates for Location' to resolve '{location_name}' to a latitude/longitude.
        2. Use 'Fetch Live Weather with Fallback' with those coordinates to get
           current temperature, cloud cover, and irradiance.
        3. Reason about whether the weather could plausibly be influencing this detection:
           - If the detection came from a sensor reading (power/electrical data): heavy
             cloud cover or low irradiance can genuinely reduce power output and look
             like a fault. Say so if that's the case.
           - If the detection came from a photo (a visible defect like dirt, cracks, or
             bird droppings): weather mostly affects image lighting/quality, not whether
             the defect is physically on the panel. Say plainly that weather doesn't
             rule the visual finding in or out, rather than forcing a link that isn't there.
        4. Decide a status: CONFIRMED_FAULT, POSSIBLE_WEATHER_INFLUENCE, or UNCERTAIN,
           with a confidence score (0.0-1.0).
        5. Write up the finding as a formal technical ticket, in the voice of a
           senior solar reliability engineer, using EXACTLY this structure:

        **PV Diagnostic Ticket**
        - Site: {site_id}
        - District: {location_name}
        - Coordinates: [Lat, Lon -- from step 1]
        - ML Model Prediction: [fault type + model confidence]

        **Weather Conditions**
        - Temperature: [°C]
        - Cloud Cover: [%]
        - Irradiance: [W/m²]

        **Diagnosis**
        - Status: [CONFIRMED_FAULT | POSSIBLE_WEATHER_INFLUENCE | UNCERTAIN]
        - Confidence: [0.0-1.0]
        - Justification: [2-3 plain-language sentences connecting the weather to the
          detection type, per the reasoning in step 3]

        **Recommended Next Step**
        - [One clear sentence: e.g. dispatch a technician for physical inspection,
          or no action needed and continue monitoring]
        """,
        expected_output="A fully filled-in technical ticket following the exact structure above, no placeholder brackets left unresolved, and no mention of theoretical power or SCADA baselines anywhere.",
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