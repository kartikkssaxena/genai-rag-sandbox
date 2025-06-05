from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from pathlib import Path
import json
from langchain_core.documents import Document 
import os

# Step 0: Config from environment
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Step 1: Load all .txt and .pdf documents from 'my_docs' folder
doc_folder = Path("my_docs")

all_docs = []
for file_path in doc_folder.glob("*"):
    if file_path.suffix.lower() == ".txt":
        loader = TextLoader(str(file_path))
        all_docs.extend(loader.load())
    elif file_path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(file_path))
        all_docs.extend(loader.load())
    elif file_path.suffix.lower() == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Example: assuming JSON contains a list of texts or a large single text blob
        if isinstance(data, list):
            for item in data:
                content = item if isinstance(item, str) else json.dumps(item)
                all_docs.append(Document(page_content=content, metadata={"source": str(file_path)}))
        elif isinstance(data, dict):
            # Flatten or pick a specific field if known
            content = json.dumps(data)  # OR data["text"] if you know the structure
            all_docs.append(Document(page_content=content, metadata={"source": str(file_path)}))

if not all_docs:
    print("No documents found in 'my_docs' folder.")
    exit(1)
print("------------------step 1 completed ---------------------")
# Step 2: Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(all_docs)

print("------------------step 2 completed ---------------------")
# Step 3: Use Ollama for embeddings (Ollama server must be running on host)
embedding_model = OllamaEmbeddings(model="llama3", base_url=OLLAMA_BASE_URL)

print("------------------step 3 completed ---------------------")
# Step 4: Connect to Qdrant
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
collection_name = "my_docs"

print("------------------step 4 completed ---------------------")
# Determine embedding vector size dynamically
vector_size = len(embedding_model.embed_query("test"))

# Step 5: Create or recreate collection in Qdrant
if qdrant.collection_exists(collection_name):
    qdrant.delete_collection(collection_name)

qdrant.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
)

print("------------------step 5 completed ---------------------")
# Step 6: Store embeddings in Qdrant
vector_store = Qdrant(
    client=qdrant,
    collection_name=collection_name,
    embeddings=embedding_model,
)

vector_store.add_documents(chunks)

print("------------------step 6 completed ---------------------")
# Step 7: Query interface
query = "give me details on leave policy and work from home guidelines ??"
results = vector_store.similarity_search(query, k=3)

print("\nTop matching results:\n")
for idx, res in enumerate(results, 1):
    print(f"{idx}. {res.page_content}\n")
