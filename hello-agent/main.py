import os

from agents import (
    Agent,
    Runner,
    RunConfig,
    OpenAIChatCompletionsModel,
    AsyncOpenAI
)
from dotenv import load_dotenv

load_dotenv()

gemini_api = os.getenv("GEMINI_API")

external_client = AsyncOpenAI(
    api_key = gemini_api,
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model = "gemini-2.0-flash",
    openai_client = external_client
)

config = RunConfig(
    model = model,
    model_provider = external_client,
    tracing_disabled = True
)

hello_agent = Agent(name = "Hello_agent" , instructions = "you are a hello agent")

result = Runner.run_sync(hello_agent , input = "Hi Agent!" , run_config = config)
print(result.final_output)