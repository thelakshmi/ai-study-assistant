def load_document(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    return text


document = load_document("notes/os.txt")

print("Document loaded successfully!")
print()
print(document)