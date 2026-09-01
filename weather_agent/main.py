from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


client = OpenAI(
    api_key= os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def main():
    user_query = input("--> ")
    response = client.chat.completions.create(
        model="gemini-3.1-flash-lite",
        messages=[
            {
                "role":"user",
                "content":user_query,
            }
        ]
    )
   
    print(response.choices[0].message.content)

main()
