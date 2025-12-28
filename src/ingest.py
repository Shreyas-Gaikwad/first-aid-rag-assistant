from pathlib import Path
from pypdf import PdfReader
import re
import json

RAW_PDF_DIR = Path("data/raw_pdfs")
OUTPUT_PATH = Path("data/extracted/clean_text.json")


def clean_text(text: str) -> str:
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def ingest_pdfs():
    documents = []

    for pdf_path in RAW_PDF_DIR.glob("*.pdf"):
        reader = PdfReader(pdf_path)
        pages = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)

        cleaned = clean_text("\n".join(pages))

        documents.append({
            "source": pdf_path.name,
            "text": cleaned
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    print(f"Ingested {len(documents)} documents")


if __name__ == "__main__":
    ingest_pdfs()