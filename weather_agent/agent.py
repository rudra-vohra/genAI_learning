from openai import OpenAI
import json
import os
import requests
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel,Field



load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


def get_weather(city: str):
    url = f"https://wttr.in/{city.lower()}?format=%C+%t"

    response = requests.get(url, timeout=10)

    if response.status_code == 200:
        return f"The weather in the {city} is {response.text}"

    return "Something went wrong"


available_tools = {
    "get_weather": get_weather,
}

class MyOutput(BaseModel):
    step: str = Field(...,description="The ID of the step. Example: PLAN,START,OUTPUT etc")
    content:Optional[str] = Field(None,description="The optional string content")
    tool:Optional[str] = Field(None,description="The ID of the tool to call")
    input:Optional[str] = Field(None,description="The input to be given to the tool")


system_prompt = """
You're an expert AI Assistant in resolving user queries.
You work in three steps:
1. START
2. PLAN
4.TOOL
5.OBSERVE
6.OUTPUT
You need to first plan what needs to be done,finally you give an OUTPUT.
You can also call a tool if required from the list of available tools. 
For every tool call wait for the observe step which is the output of the called tool

Rules:
1. Return EXACTLY ONE JSON object in every response.
2. NEVER return a JSON array.
3. NEVER return multiple JSON objects in one response.
4. Do not use markdown or code fences.
5. The sequence must be:
   START -> PLAN -> PLAN -> ... ->TOOL-> OUTPUT
6. Strictly follow the given JSON output format
7. After returning START or PLAN, wait for the next user message before generating the next step.
8. When the user message is "Continue to the next step.", continue to the next step.
9. Never generate the next step automatically in the same response.


Available tools :
- get_weather(city: str): Takes city name as input string and return weather info about the city 

CRITICAL:
You are allowed to return ONLY ONE JSON object per response.
After returning one JSON object, STOP GENERATING.
Do NOT return the next step in the same response.


Example:

User:
What is the weather of delhi?

User:
{"step":"Start","content":"What is the weather of delhi?"}
Assistant:
{"step":"PLAN","content":"Seems like the user is interested in knowing the weather of Delhi"}
Assistant:
{"step":"PLAN","content":"Lets see if we have any available tools from the list of tools"}
Assistant:
{"step":"PLAN","content":"Great we have the get_weather tool available for this query"}
Assistant:
{"step":"PLAN","content":"I need to call get_weather tool with delhi as input for the city"}
Assistant:
{"step":"TOOL","tool":"get_weather","input":"delhi"}
Assistant:
{"step":"OBSERVE","tool":"get_weather","output":"The temperature of delhi is cloudy with 20 C"}
Assistant:
{"step":"PLAN","content":"Great I got the weather info about delhi"}
Assistant:
{"step":"OUTPUT","content":"The current weather in delhi is 20 C with some cloudy sky."}


The output JSON format MUST be exactly:

{
    "step": "START",
    "content": "string"
}

OR

{
    "step": "PLAN",
    "content": "string"
}

OR

{
    "step": "OUTPUT",
    "content": "string"
}

OR

{
    "step": "TOOL",
    "tool": "string",
    "input": "string"
}

OR

{
    "step": "OBSERVE",
    "tool": "string",
    "input": "string",
    "output": "string"
}
"""

message_history = [
    {
        "role": "system",
        "content": system_prompt
    }
]

while True:
    user_query = input("enter your question -> ")
    message_history.append({
        "role": "user",
        "content": user_query
    })
    while True:
        response = client.chat.completions.parse(
            model="gemini-3.1-flash-lite",
            response_format=MyOutput,
            messages=message_history    
        )

        choice = response.choices[0]
        raw_result = choice.message.content

        if raw_result is None:
            print("❌ Gemini returned no content.")
            print("finish_reason:", choice.finish_reason)
            print("Full response:", response)
            break

        message_history.append({
            "role": "assistant",
            "content": raw_result
        })

        parsed_result = response.choices[0].message.parsed
        if parsed_result.step == "START":

            print("🔥", parsed_result.content)

            message_history.append({
                "role": "user",
                "content": "Continue to the next step."
            })
            continue

        if parsed_result.step == "PLAN":

            print("🧠", parsed_result.content)

            message_history.append({
                "role": "user",
                "content": "Continue to the next step."
            })
            continue

        if parsed_result.step == "TOOL":

            tool_to_call = parsed_result.tool
            tool_input = parsed_result.input

            print(f"🛠️: {tool_to_call}({tool_input})")

            if tool_to_call not in available_tools:
                print(f"❌ Unknown tool: {tool_to_call}")
                break

            tool_response = available_tools[tool_to_call](tool_input)

            print(
                f"🛠️: {tool_to_call}({tool_input}) : {tool_response}"
            )
            message_history.append({
                "role": "user",
                "content": json.dumps({
                    "step": "OBSERVE",
                    "tool": tool_to_call,
                    "input": tool_input,
                    "output": tool_response
                })
            })
            continue

        if parsed_result.step == "OUTPUT":
            print("🤖", parsed_result.content)
            break
