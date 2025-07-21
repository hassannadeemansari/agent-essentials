import os

from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    handoff,
)
from dotenv import load_dotenv
from agents.run import RunConfig

load_dotenv()


gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("api key is not set , esure that your key is defined inside .env.file!")
    
external_client = AsyncOpenAI(
    api_key = gemini_api_key,
    base_url = "https://generativelanguage.googleapis.com/v1beta"
)

model = OpenAIChatCompletionsModel(
    model = "gemini-2.0-flash",
    openai_client = external_client,
)

config = RunConfig(
    model = model,
    model_provider = external_client,
    tracing_disabled=True
)



french_agent = Agent(
    name = "French",
    instructions = "you translate only the user sentence or word from english to french "
)

spanish_agent = Agent(
    name = "Spanish",
    instructions = "you translate only the user sentence or word from english to Spanish"
)

arabic_agent = Agent(
    name = "Arabic",
    instructions = "you translate only the user sentence or word from english to Arabic"
)

main_agent = Agent(
    name = "main_agent",
    instructions = """you are a translation you use the tools that given to you,
    if user ask multiple tranclation your gve it to its exoert agent or tool, if user ask other question or other languge which tool sre not 
    available so you denied it dont give answere by your self """,
    tools = [
        french_agent.as_tool(
            tool_name = "translate_to_French",
            tool_description = "translate the user message from english to french"                
        ),
        spanish_agent.as_tool(
            tool_name = "translate_to_spanish",
            tool_description = "translate the user message from english to Spanish" 
        ),
        arabic_agent.as_tool(
            tool_name = "translate_to_arabic",
            tool_description = "translate the use message from english to Arabic"
        )
    ]
)






# french_agent = Agent(
#     name="French",
#     instructions="Translate only the user sentence or word from English to French."
# )

# spanish_agent = Agent(
#     name="Spanish",
#     instructions="Translate only the user sentence or word from English to Spanish."
# )

# arabic_agent = Agent(
#     name="Arabic",
#     instructions="Translate only the user sentence or word from English to Arabic."
# )

# main_agent = Agent(
#     name="main_agent",
#     instructions="""
#     You are a translation agent. Use the tools provided to you.
#     If the user asks for translation into French, Spanish, or Arabic, use the corresponding tool.
#     If the user asks for multiple translations, delegate to the respective agents.
#     Do NOT answer by yourself.
#     """,
#     tools=[
#         french_agent.as_tool(
#             tool_name="translate_to_french",
#             tool_description="Translate the user message from English to French."
#         ),
#         spanish_agent.as_tool(
#             tool_name="translate_to_spanish",
#             tool_description="Translate the user message from English to Spanish."
#         ),
#         arabic_agent.as_tool(
#             tool_name="translate_to_arabic",
#             tool_description="Translate the user message from English to Arabic."
#         )
#     ]
# )




    

user = input("user :")
Response = Runner.run_sync(main_agent, user  ,run_config = config)
print(Response.final_output)