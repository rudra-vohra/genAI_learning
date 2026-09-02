import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

client = OpenAI(
    api_key= os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Google Gemini embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="rag",
    embedding=embeddings,
)

def process_query(query: str):
    print("Searchig query ",query)
    search_results = vector_db.similarity_search(query)
    context = "\n\n".join([f"Page Content: {result.page_content}\nPage Number:{result.metadata['page_label']}\nFile location: {result.metadata['source']}" for result in search_results])

    system_prompt = """
        You are a helpful AI assistant who answers user query based on the available context retrieved from a pdf file along with page number and page_contents

        You should only answer the user based on the following context and navigate the user to open the right page number to know more 
        
        {context}
    """
    response = client.chat.completions.create(
        model="gemini-3.5-flash-lite",
        messages=[
        {
            "role" : "system",
            "content": system_prompt
        },
        {
                "role" : "user",
                "content" : query
        }
        ]
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content
