import streamlit as st
import os
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, RunContextWrapper
from agents.tool import function_tool
from agents.run import RunConfig
from typing import Optional, List, Dict
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

run_config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)

class UserSessionContext(BaseModel):
    name: str
    uid: int
    goal: Optional[dict] = None
    diet_preferences: Optional[str] = None
    workout_plan: Optional[dict] = None
    meal_plan: Optional[List[str]] = None
    injury_notes: Optional[str] = None
    handoff_logs: List[str] = []
    progress_logs: List[Dict[str, str]] = []

@function_tool
def GoalAnalyzerTool() -> str:
    return "Converts the user's goals into a structured format."

@function_tool
def MealPlannerTool() -> str:
    return "Provides a 7-day diet plan based on preferences."

@function_tool
def WorkoutRecommenderTool() -> str:
    return "Suggests workouts according to the user's fitness level."

@function_tool
def CheckinSchedulerTool() -> str:
    return "Schedules weekly progress checks."

@function_tool
def ProgressTrackerTool() -> str:
    return "Logs and updates the user's progress."

@function_tool
def InjuryAdviceTool() -> str:
    return "Avoid weight-bearing exercises. Focus on upper body workouts."

@function_tool
def DiabeticDietTip() -> str:
    return "Low-carb, high-fiber meals recommended. Avoid sugary foods."

first_agent = Agent(
    name="EscalationAgent",
    instructions="When the user wants to talk to a real human trainer.",
    tools=[],
    handoff_description="Escalating your request to a human trainer as per your preference."
)

second_agent = Agent(
    name="NutritionExpertAgent",
    instructions="Provide advice for users with dietary conditions like diabetes.",
    tools=[DiabeticDietTip],
    handoff_description="This request requires specialized nutritional advice."
)

third_agent = Agent(
    name="InjurySupportAgent",
    instructions="Help users with injuries by giving special fitness advice.",
    tools=[InjuryAdviceTool],
    handoff_description="This case involves a physical concern."
)

triage_agent = Agent(
    name="triage_agent",
    instructions="You are a health and wellness assistant. You analyze goals, provide plans, and hand off to specialists if needed.",
    tools=[
        GoalAnalyzerTool,
        MealPlannerTool,
        WorkoutRecommenderTool,
        CheckinSchedulerTool,
        ProgressTrackerTool
    ],
    handoffs=[first_agent, second_agent, third_agent]
)

st.set_page_config(page_title="Neuro Bot", layout="centered")
st.title("🧠 Neuro Bot")

user_input = st.text_area("Enter your health goal:", height=150)

if st.button("Submit") and user_input:
    with st.spinner("Analyzing your health goals..."):
        user_context = UserSessionContext(name="User", uid=1)
        context_wrapper = RunContextWrapper(user_context)

        result = Runner.run_sync(
            agent=triage_agent,
            input=user_input,
            context=context_wrapper,
            config=run_config
        )

        st.markdown("### 🩺 AI Response:")
        st.success(result.final_output)
