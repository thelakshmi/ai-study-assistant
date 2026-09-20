from document_loader import load_document
from chunker import split_into_chunks
from sentence_transformers import SentenceTransformer


# 1. Load the document
document = load_document("notes/os.txt")

# 2. Split the document into chunks
chunks = split_into_chunks(document)

# 3. Load an embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Convert each chunk into an embedding
embeddings = model.encode(chunks)

print("Number of chunks:", len(chunks))
print("Number of embeddings:", len(embeddings))

# Show the first embedding
print("\nFirst embedding:")
print(embeddings[0])

print("\nEmbedding dimensions:", len(embeddings[0]))