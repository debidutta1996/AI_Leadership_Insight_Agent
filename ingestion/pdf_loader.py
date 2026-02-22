import os
from pypdf import PdfReader

def load_pdfs(directory_path):
    documents = []

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(directory_path, filename)

            try:
                reader = PdfReader(filepath)

                full_text = ""
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"

                if full_text.strip():
                    documents.append({
                        "id": filename,
                        "text": full_text,
                        "metadata": {"source": filename}
                    })
                    print("Loaded:", filename)
                else:
                    print(f"No extractable text in {filename}")

            except Exception as e:
                print(f"Failed to read {filename}: {e}")

    return documents