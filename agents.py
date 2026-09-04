import json
from crewai import Agent, Task, Crew, Process, LLM
from tools import get_coordinates_for_location, fetch_weather_data

def run_multi_agent_pipeline(prediction: dict, site_id: str = "SITE_01", location_name: str = "Kilinochchi", llm: LLM = None) -> str:

    # The LSTM path predicts a future failure date from years of historical
    # degradation trend (Health_Indicator decay), not from a single point-in-time
    # sensor/image reading. Today's weather has no bearing on that: a long-run
    # degradation slope isn't explained by current cloud cover, and pretending
    # otherwise ("similar low-irradiance conditions could have existed back when
    # the trend started") is just a guess dressed up as reasoning. So this data
    # type skips the weather tools entirely and goes straight to reporting the
    # trend finding.
    if str(prediction.get("data_type", "")).startswith("Time-Series"):
        return _run_trend_only_pipeline(prediction, site_id, location_name, llm)
    return _run_weather_checked_pipeline(prediction, site_id, location_name, llm)


def _run_weather_checked_pipeline(prediction: dict, site_id: str, location_name: str, llm: LLM = None) -> str:
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


def _run_trend_only_pipeline(prediction: dict, site_id: str, location_name: str, llm: LLM = None) -> str:
    # ----------------------------------------------------------------------
    # AGENT: DEGRADATION TREND ANALYST
    # (no tools -- this data type is a long-horizon RUL projection from the
    #  LSTM model, not a point-in-time reading, so there's nothing for
    #  weather to plausibly explain. Just interpret the trend and write it up.)
    # ----------------------------------------------------------------------
    trend_expert = Agent(
        role="Senior PV Degradation Analyst",
        goal="Interpret an LSTM remaining-useful-life projection for a PV array and issue a clear diagnostic ticket, without reaching for explanations the data doesn't support.",
        backstory="Senior Solar Reliability Engineer with 15+ years reading long-run degradation trends from PV health-indicator time series, careful never to attribute a multi-year trend to a single day's weather.",
        tools=[],
        llm=llm,
        verbose=True
    )

    task_ticket = Task(
        description=f"""
        Site '{site_id}' is located in the '{location_name}' district of Sri Lanka.
        The ML model output is a long-run degradation trend projection: {json.dumps(prediction)}

        This came from an LSTM model trained on the site's historical Health_Indicator
        time series, projecting forward. It is NOT a point-in-time sensor or image
        reading, so do not check or mention current weather, cloud cover, or
        irradiance anywhere in the ticket -- a multi-year degradation slope isn't
        explained by today's conditions, and pretending it might be is just a guess
        dressed up as reasoning.

        Do the following, in order:
        1. Decide a status: CONFIRMED_FAULT (a failure date is projected within
           the horizon) or NOMINAL (no failure projected within the horizon).
        2. Assign a confidence score (0.0-1.0) reflecting how clear-cut the trend is.
        3. Write up the finding as a formal technical ticket, in the voice of a
           senior solar reliability engineer, using EXACTLY this structure:

        **PV Diagnostic Ticket**
        - Site: {site_id}
        - District: {location_name}
        - Data Type: Time-Series Sensor Log (LSTM degradation trend)
        - ML Model Prediction: [detection + current health indicator, from the data above]

        **Diagnosis**
        - Status: [CONFIRMED_FAULT | NOMINAL]
        - Confidence: [0.0-1.0]
        - Justification: [2-3 plain-language sentences interpreting the degradation
          trend itself -- current health indicator, trajectory, and projected
          failure date if any. No weather, no SCADA baseline.]

        **Recommended Next Step**
        - [One clear sentence: e.g. schedule preventive maintenance ahead of the
          projected date, or no action needed and continue monitoring]
        """,
        expected_output="A fully filled-in technical ticket following the exact structure above, no placeholder brackets left unresolved, and no mention of weather, cloud cover, irradiance, theoretical power, or SCADA baselines anywhere.",
        agent=trend_expert
    )

    # ----------------------------------------------------------------------
    # CREW
    # ----------------------------------------------------------------------
    crew = Crew(
        agents=[trend_expert],
        tasks=[task_ticket],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()
    return str(result)