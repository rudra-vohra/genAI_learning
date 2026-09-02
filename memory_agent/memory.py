import os
import json
from mem0 import Memory
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key= GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
# mem0 is an AI long term memory layer designed for LLM's
config = {
    "version": "v1.1",
    "embedder":{
        "provider":"gemini",
        "config":{
            "api_key":GEMINI_API_KEY,
            "model":"gemini-embedding-001",
            "embedding_dims": 1536
        }
    },
    "llm":{
        "provider":"gemini",
        "config":{
            "api_key":GEMINI_API_KEY,
            "model":"gemini-3.5-flash-lite"
        }
    },
    "graph_store":{
        "provider":"neo4j",
        "config":{
            "url": os.getenv("NEO_CONNECTION_URI"),
            "username":os.getenv("NEO_USERNAME"),
            "password":os.getenv("NEO_PASSWORD")
        }
    },
    "vector_store":{
        "provider":"qdrant",
        "config":{
            "host": "localhost",
            "port": 6333,
            "embedding_model_dims": 1536
        }
    }
}

memory_client = Memory.from_config(config)

# The new mem0 does not require the above config and below old code 
# the new mem0 method of adding and searching memories:-

# Memory client
# memory = MemoryClient( api_key=os.getenv("MEM0_API_KEY") )

# memory searching and fromatting
# results = memory.search(
#     user_query,
#     user_id="john"
#)
# memories = [ f"Memory: {mem.get('memory')}" for mem in search_memory.get("results", []) ]

# memory adding
# memory.add(
#     [
#         {"role": "user", "content": user_query},
#         {"role": "assistant", "content": ai_response}
#     ],
#     user_id="john"
# )

# the new mem0 architecture does not require explicit declaration of vectorDb and graphDb it uses its own memory architecture which includes vector search ,keyword matching and entity matching for searching which is better than just string chunks at the vectorDB and search for semantics at the time of retrieval

 

while True:
    user_query = input("-> ")

    search_memory = memory_client.search(
        query=user_query,
        filters={"user_id": "john"}
    )

    memories = [f"Id: {mem.get("id")}\n Memory: {mem.get("memory")}" for mem in search_memory.get("results")]

    system_prompt = f"""
            Here's the context about the user:
            {json.dumps(memories)}
    """

    response = client.chat.completions.create(
        model="gemini-3.5-flash-lite",
        messages=[
            {
                "role":"system",
                "content":system_prompt
            },
            {
                "role":"user",
                "content":user_query
            }
        ]
    )

    ai_response = response.choices[0].message.content
    print(ai_response)

    memory_client.add(
        user_id="john",
        messages=[
            {
                "role":"user",
                "content":user_query
            },
            {
                "role":"assistant",
                "content":ai_response
            }
        ]
    )
    print("memory saved")

