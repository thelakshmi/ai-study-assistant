import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN")
)

print("AI Study Assistant")
print("Type 'quit' to exit.\n")

while True:
    question = input("You: ")

    if question.lower() == "quit":
        print("Goodbye!")
        break

    if not question.strip():
        continue

    response = client.responses.create(
        model="openai/gpt-oss-120b:groq",
        instructions="You are a helpful AI study assistant.",
        input=question
    )

    print("AI:", response.output_text)
    print()