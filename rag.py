import os
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Connect to Hugging Face
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN")
)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Chroma
chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_collection(
    name="os_documents"
)

# Store conversation history
conversation_history = []

print("AI Study Assistant")
print("Ask questions about your study material.")
print("Type 'quit' to exit.\n")

while True:

    # Get question from user
    question = input("You: ")

    if question.lower() == "quit":
        print("Goodbye!")
        break

    if not question.strip():
        continue

    # Convert question into embedding
    question_embedding = embedding_model.encode(question).tolist()

    # Retrieve relevant chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )

    # Get retrieved source text
    retrieved_documents = results["documents"][0]

    # Get IDs of retrieved chunks
    retrieved_ids = results["ids"][0]

    # Combine retrieved chunks
    context = "\n\n".join(retrieved_documents)

    # Build conversation history
    history = ""

    for previous_question, previous_answer in conversation_history:
        history += f"""
User: {previous_question}
AI: {previous_answer}

"""

    # Create grounded prompt
    prompt = f"""
Use the following context from the user's study material
to answer the current question.

You may use the conversation history to understand
follow-up questions.

If the answer is not present in the provided context,
say:
"I don't know based on the provided document."

Conversation history:
{history}

Current context:
{context}

Current question:
{question}
"""

    # Ask the LLM
    response = client.responses.create(
        model="openai/gpt-oss-120b:groq",
        instructions=(
            "You are a helpful AI study assistant. "
            "Answer using only the provided study material."
        ),
        input=prompt
    )

    answer = response.output_text

    # Display answer
    print("\nAI:", answer)

    # Display sources
    print("\nSources used:")

    for chunk_id, document in zip(retrieved_ids, retrieved_documents):

        # Convert chunk_0 → Chunk 1
        chunk_number = int(chunk_id.split("_")[1]) + 1

        print(f"\n--- Chunk {chunk_number} ---")
        print(document)

    # Save this conversation turn
    conversation_history.append((question, answer))

    # Keep only the last 5 question-answer pairs
    conversation_history = conversation_history[-5:]

    print()