from document_loader import load_document


def split_into_chunks(text, chunk_size=500, overlap=100):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


document = load_document("notes/os.txt")

chunks = split_into_chunks(document)

print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)