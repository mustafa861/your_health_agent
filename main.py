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

first_agent = Agent(
    name="EscalationAgent",
    instructions="When the user wants to talk to a real human trainer.",
    handoff_description="Escalating your request to a human trainer as per your preference.and when traige agent say you you also give me answer"
)

second_agent = Agent(
    name="NutritionExpertAgent",
    instructions="When the user has complex diet issues like diabetes or food allergies. and when traige agent say you you also give me answer",
    handoff_description="This request requires specialized nutritional advice. Handing off to a qualified expert."
)

third_agent = Agent(
    name="InjurySupportAgent",
    instructions="When the user has an injury or a physical issue. and when traige agent say you you also give me answer",
    handoff_description="This case involves a physical concern. Forwarding to a medical support expert for further help."
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

context = UserSessionContext(
    name="TestUser",
    uid=123
)

wrapped_context = RunContextWrapper(context)

triage_agent = Agent(
    name="triage_agent",
    instructions="you are a health agent and you help the patient and where you are confuse you can handoff the agent and say them that theycan give me answer and you can use tool",
    tools=[
        GoalAnalyzerTool,
        MealPlannerTool,
        WorkoutRecommenderTool,
        CheckinSchedulerTool,
        ProgressTrackerTool,
    ],
    handoffs=[first_agent, second_agent, third_agent]
)

result = Runner.run_sync(
    triage_agent,
    input="I'm 45 years old, I have type 2 diabetes, and recently injured my ankle. I want a weight loss plan tailored to my condition. Can you suggest a customized workout and diet? yes i like that you transfer to speacilist",
    context=wrapped_context,
    run_config=run_config
)

print(result.final_output)
