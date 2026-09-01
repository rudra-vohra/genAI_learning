import json
import os
import requests
import wave
import base64
import asyncio
import speech_recognition as sr
import subprocess
import winsound
from google import genai
from dotenv import load_dotenv
from typing import Optional
from openai import OpenAI
from pydantic import BaseModel,Field
load_dotenv()

# The folder this script lives in — the only place file-creating commands can act
SANDBOX_DIR = os.path.dirname(os.path.abspath(__file__))

BLOCKED_PATTERNS = [
    "..",           # parent directory traversal
    "~",            # home directory shortcut
    ":/",           # just in case of forward-slash drive refs
    "\\windows",    # common Windows system paths
    "/etc", "/bin", "/usr", "/System",  # common *nix/mac system paths
    "format ", "diskpart", "shutdown", "rd /s", "rmdir /s", "del /s",
]


client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


tts_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

def save_audio(filename, audio_data):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(audio_data)

async def text_to_speech(text):
    # generating speech
    interaction = await tts_client.aio.interactions.create(
        model="gemini-3.1-flash-tts-preview",

        input=f"""
        Say naturally and cheerfully.:

        {text}
        """,

        response_format={
            "type": "audio"
        },

        generation_config={
            "speech_config": [
                {
                    "voice": "Orus"
                }
            ]
        }
    )

    # Gemini returns Base64 encoded audio
    audio_data = base64.b64decode(
        interaction.output_audio.data
    )

    save_audio(
        "out.wav",
        audio_data
    )

    print("Speech generated → out.wav")
    winsound.PlaySound(
        "out.wav",
        winsound.SND_FILENAME
    )

def run_command(cmd: str):
    lowered = cmd.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern in lowered:
            return f"❌ Command rejected: contains disallowed pattern '{pattern}'."

    try:
        result = subprocess.run(
            cmd,
            shell=True,           # needed for Windows builtins (mkdir, echo, dir, etc.)
            cwd=SANDBOX_DIR,      # always executes inside the script's own folder
            capture_output=True,
            text=True,
            timeout=15
        )
        output = result.stdout + result.stderr
        return output.strip() if output.strip() else f"Command exited with code {result.returncode}"
    except subprocess.TimeoutExpired:
        return "❌ Command timed out."

def get_weather(city: str):
    url = f"https://wttr.in/{city.lower()}?format=%C+%t"

    response = requests.get(url, timeout=10)

    if response.status_code == 200:
        return f"The weather in the {city} is {response.text}"

    return "Something went wrong"

def write_file(filename: str, content: str):
    path = os.path.abspath(os.path.join(SANDBOX_DIR, filename))

    # This is the safety ste as we want the llm to only manipulate thing inside our curret directory , so we add a safety step."does the final resolved path still begin with my sandbox folder's path?"
    # If yes → the file is safely inside your allowed folder → continue.
    # If no → someone (or the LLM) tried to point the file somewhere outside your folder → reject it immediately, and return an error message string instead of writing anything.

    if not path.startswith(SANDBOX_DIR):
        return f"❌ Path '{filename}' resolves outside the allowed folder."
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Wrote {len(content)} characters to {filename}"
    except Exception as e:
        return f"❌ Failed to write file: {e}"

    # Full flow of the wite_file method
    # Say the LLM calls:
    # write_file(filename="index.html", content="<html><body>Hi</body></html>"):
    # 1. Python builds the full safe path: ...\12_voice_agent\index.html
    # 2. Checks it's inside your folder ✅
    # 3. Opens that file in write mode
    # 4. Writes <html><body>Hi</body></html> into it, exactly as given
    # 5. Returns "Wrote 30 characters to index.html"
    # 6. That confirmation then gets shown back to the LLM as the OBSERVE result, so it knows the file was created successfully.
available_tools = {
    "get_weather": get_weather,
    "run_command": run_command,
    "write_file": write_file,
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
- run_command(cmd: str): Takes a system command as string and executes it. Use ONLY for structural operations like mkdir, dir, ping — NEVER for writing HTML/CSS/JS file content, since shell characters like < > " ' will break.
- write_file(filename: str, content: str): Writes text content directly into a file inside the allowed folder. ALWAYS use this for creating or overwriting files with HTML, CSS, JS, or any text content. To call it, set "input" to a JSON string like: {"filename": "index.html", "content": "<html>...</html>"}

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
async def main():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        r.pause_threshold = 4

        while True:
            print("Do you want to Speak or Type? (S/T)")
            user_choice = input("Enter your choice(S/T)->").upper()

            if user_choice == 'S':
                print("speak something -> ")
                audio = r.listen(source)
                print("Processing audio..")
                user_query = r.recognize_google(audio)
            elif user_choice == 'T':
                user_query = input("Type ur query ->")
            else:
                print("Invalid Input")
                break

            message_history.append({"role": "user","content": user_query})
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

                    # write_file needs two args (filename, content), so its input
                    # arrives as a JSON string like {"filename": "...", "content": "..."}
                    # get_weather/run_command just take a plain string, so fall back to that
                    try:
                        parsed_args = json.loads(tool_input)
                        if isinstance(parsed_args, dict):
                            tool_response = available_tools[tool_to_call](**parsed_args)
                        else:
                            tool_response = available_tools[tool_to_call](tool_input)
                    except (json.JSONDecodeError, TypeError):
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
                    # here, the below command is commented to be used as optional command if u want ur llm to speak out its output everytime a task is done. I commented it because i'm using gemini's free tts model so it exhausts easily and causes my program to break
                    # await text_to_speech(parsed_result.content)
                    break

if __name__ == "__main__":
    asyncio.run(main())

