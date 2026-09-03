import json
from crewai import Agent, Task, Crew, LLM
from tools import (
    fetch_weather_data, 
    calculate_expected_pv_power,
    get_power_baseline, 
    get_repair_procedure,
    check_parts_inventory,
    calculate_repair_roi
)

def run_multi_agent_pipeline(prediction: dict, site_id: str = "SITE_01", lat: float = 37.77, lon: float = -122.41, llm: LLM = None) -> str:
    
    # ----------------------------------------------------------------------
    # AGENT 1: VALIDATOR
    # ----------------------------------------------------------------------
    validator = Agent(
        role="PV Environmental & Physics Validator",
        goal="Validate ML predictions using weather, theoretical physics, and SCADA baselines.",
        tools=[fetch_weather_data, calculate_expected_pv_power, get_power_baseline],
        llm=llm,
        verbose=True
    )

    task_validate = Task(
        description=f"""
        1. Fetch current weather for Lat: {lat}, Lon: {lon}.
        2. Calculate theoretical power using 'Calculate Expected PV Power Output'.
        3. Query historical baseline for site '{site_id}'.
        4. Compare ML Prediction ({json.dumps(prediction)}) against empirical/theoretical baselines.
        5. Output status: CONFIRMED_FAULT, WEATHER_FALSE_POSITIVE, or LOW_CONFIDENCE with confidence score (0.0 - 1.0).
        """,
        expected_output="Validation status, justification, and confidence score.",
        agent=validator
    )

    # ----------------------------------------------------------------------
    # AGENT 2: TECHNICAL EXPERT
    # ----------------------------------------------------------------------
    tech_expert = Agent(
        role="PV Technical & Supply Chain Specialist",
        goal="Determine IEC standard repair procedures and check part availability.",
        tools=[get_repair_procedure, check_parts_inventory],
        llm=llm,
        verbose=True
    )

    task_tech = Task(
        description="""
        Check Agent 1 validation output:
        - If validation confidence < 0.50: Output "INSUFFICIENT DATA - Request thermal imaging scan".
        - If fault is confirmed: Query 'Get Deterministic Repair Procedure' and 'Check Warehouse Parts Inventory'.
        Provide exact repair steps, safety precautions, and stock readiness.
        """,
        expected_output="SOP repair steps, required tools/PPE, and warehouse stock status.",
        agent=tech_expert,
        context=[task_validate]
    )

    # ----------------------------------------------------------------------
    # AGENT 3: FINANCIAL PLANNER
    # ----------------------------------------------------------------------
    financial_planner = Agent(
        role="Solar Asset Operations & Financial Planner",
        goal="Calculate daily financial loss and optimize repair dispatch urgency.",
        tools=[calculate_repair_roi],
        llm=llm,
        verbose=True
    )

    task_financial = Task(
        description="""
        Use 'Calculate Repair ROI and Emergency Dispatch' to compute financial revenue loss ($/day).
        Combine the outputs from Agent 1 (Validation) and Agent 2 (Parts/SOP) to recommend dispatch strategy:
        EMERGENCY_DISPATCH, SCHEDULED_MAINTENANCE, or HOLD_FOR_PARTS.
        """,
        expected_output="Financial loss breakdown ($/day), breakeven calculation, and final operational dispatch recommendation.",
        agent=financial_planner,
        context=[task_validate, task_tech]
    )

    # ----------------------------------------------------------------------
    # CREW ORCHESTRATION
    # ----------------------------------------------------------------------
    crew = Crew(
        agents=[validator, tech_expert, financial_planner], 
        tasks=[task_validate, task_tech, task_financial], 
        verbose=True
    )
    
    return str(crew.kickoff())