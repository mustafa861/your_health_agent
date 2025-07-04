from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner
from agents.tool import function_tool
from agents.run import RunConfig
from dotenv import load_dotenv, find_dotenv
import os
load_dotenv(find_dotenv())

gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client =  AsyncOpenAI(
    api_key = gemini_api_key,
    base_url= "https://generativelanguage.googleapis.com/v1beta/openai/"
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
def GoalAnalyzerTool():
    return "Converts the user's goals into a structured format."

@function_tool
def MealPlannerTool():
    return "Provides a 7-day diet plan based on preferences."

@function_tool
def WorkoutRecommenderTool():
    return "Suggests workouts according to the user's fitness level."

@function_tool
def CheckinSchedulerTool():
    return "Schedules weekly progress checks."

@function_tool
def ProgressTrackerTool():
    return "Logs and updates the user's progress."


first_agent = Agent(
    name="EscalationAgent", 
   instructions="When the user wants to talk to a real human trainer.",
   handoff_description="Escalating your request to a human trainer as per your preference."

)

second_agent = Agent(
    name="NutritionExpertAgent",
    instructions="When the user has complex diet issues like diabetes or food allergies.",
    handoff_description="This request requires specialized nutritional advice. Handing off to a qualified expert."
)

third_agent=Agent(
    name="InjurySupportAgent",
    instructions="When the user has an injury or a physical issue.",
    handoff_description="This case involves a physical concern. Forwarding to a medical support expert for further help."
)

triage_agent = Agent(
    name="triage_agent",
    instructions="You have the power to delegate tasks to other agents and they give me answer.",
    handoffs=[first_agent, second_agent, third_agent]
)

result = Runner.run_sync(triage_agent, "I'm 45 years old, I have type 2 diabetes, and recently injured my ankle. I want a weight loss plan tailored to my condition. Can you suggest a customized workout and diet?", run_config=run_config)
print(result.final_output)   