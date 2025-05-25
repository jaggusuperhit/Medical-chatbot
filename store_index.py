from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure Pinecone
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Step 1: Load and process PDF data
print("Loading PDF files...")
extracted_data = load_pdf_file(data='Data/')
print(f"Loaded {len(extracted_data)} documents")

# Step 2: Split text into chunks
print("Splitting text into chunks...")
text_chunks = text_split(extracted_data)
print(f"Created {len(text_chunks)} text chunks")

# Step 3: Initialize embeddings
print("Downloading embeddings...")
embeddings = download_hugging_face_embeddings()

# Step 4: Initialize Pinecone
print("Initializing Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Step 5: Create Pinecone index
index_name = "medicalbot"
print(f"Creating index '{index_name}'...")

# Check if index exists first to avoid errors
existing_indexes = pc.list_indexes().names()
if index_name in existing_indexes:
    print(f"Index '{index_name}' already exists. Deleting it first...")
    pc.delete_index(index_name)

pc.create_index(
    name=index_name,
    dimension=384,  # Must match the embedding model's dimension
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

# Wait for index to be ready
import time
while not pc.describe_index(index_name).status['ready']:
    print("Waiting for index to be ready...")
    time.sleep(5)

# Step 6: Store embeddings in Pinecone
print("Storing embeddings in Pinecone...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)

print("Indexing complete!")