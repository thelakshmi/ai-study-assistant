import chromadb

from document_loader import load_document
from chunker import split_into_chunks
from sentence_transformers import SentenceTransformer


# Load document
document = load_document("notes/os.txt")

# Split document into chunks
chunks = split_into_chunks(document)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
embeddings = model.encode(chunks).tolist()


# Connect to Chroma
client = chromadb.PersistentClient(path="./chroma_db")


# Delete old collection if it exists
try:
    client.delete_collection(name="os_documents")
except Exception:
    pass


# Create a fresh collection
collection = client.create_collection(
    name="os_documents"
)


# Store chunks and embeddings
collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings
)


print("Vector database updated successfully!")
print("Number of chunks stored:", collection.count())