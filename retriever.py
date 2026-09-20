import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection(
    name="os_documents"
)

question = "What is an operating system?"

question_embedding = model.encode(question).tolist()

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3,
    include=["documents", "distances"]
)

print("Question:")
print(question)

print("\nRelevant chunks:\n")

for i, (document, distance) in enumerate(
    zip(results["documents"][0], results["distances"][0])
):

    print(f"--- Result {i + 1} ---")
    print(f"Distance: {distance:.4f}")
    print(document)
    print()