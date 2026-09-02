import time
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore


# Load environment variables
load_dotenv()


# PDF path
path = Path(__file__).parent / "nodejs.pdf"


# Load PDF
loader = PyPDFLoader(file_path=path)
docs = loader.load()


# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400
)

chunks = text_splitter.split_documents(docs)


# Google Gemini embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)

# Throttling settings
batch_size = 10
delay = 10  # seconds


first_batch = chunks[:batch_size]

vector_store = QdrantVectorStore.from_documents(
    documents=first_batch,
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name="rag"
)

# Add remaining batches
for i in range(batch_size, len(chunks), batch_size):

    batch = chunks[i:i + batch_size]

    vector_store.add_documents(batch)

    processed = min(i + batch_size, len(chunks))

    print(f"Processed {processed}/{len(chunks)} chunks")

    # Throttle API requests
    if processed < len(chunks):
        print(f"Waiting {delay} seconds...")
        time.sleep(delay)

print("Indexing of documents done!")

