import json
from pathlib import Path

INPUT_PATH = Path("data/extracted/clean_text.json")
OUTPUT_PATH = Path("data/extracted/chunks.json")

MAX_WORDS = 320
OVERLAP_WORDS = 40


def chunk_text(text):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        words = para.split()
        wlen = len(words)

        if wlen > MAX_WORDS:
            for i in range(0, wlen, MAX_WORDS - OVERLAP_WORDS):
                chunks.append(" ".join(words[i:i + MAX_WORDS]))
            current, current_len = [], 0
            continue

        if current_len + wlen <= MAX_WORDS:
            current.append(para)
            current_len += wlen
        else:
            chunks.append(" ".join(current))
            overlap = " ".join(" ".join(current).split()[-OVERLAP_WORDS:])
            current = [overlap, para]
            current_len = len(overlap.split()) + wlen

    if current:
        chunks.append(" ".join(current))

    return chunks


def build_chunks():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)

    all_chunks = []

    for doc in documents:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{doc['source']}_{i}",
                "source": doc["source"],
                "text": chunk
            })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Created {len(all_chunks)} chunks")


if __name__ == "__main__":
    build_chunks()