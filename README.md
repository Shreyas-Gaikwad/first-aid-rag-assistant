# First Aid RAG Assistant (Offline)

An end-to-end, **offline-first retrieval-augmented generation (RAG) system** for first-aid guidance, built with a strong focus on **safety, determinism, and local inference**.

This project goes beyond a basic chatbot by implementing a full RAG pipeline: PDF knowledge ingestion, semantic chunking, dense vector retrieval with FAISS, and a **strictly constrained local LLM** to reduce hallucinations in a medical-adjacent domain.

The system is designed to run **entirely offline**, making it suitable for low-connectivity or privacy-sensitive environments.

<img width="1907" height="677" alt="image" src="https://github.com/user-attachments/assets/c3dc11c6-6600-4cc1-b5c9-27b267b64904" />

**Tech Stack:**
Python · PyTorch · Hugging Face Transformers · FAISS · LLaMA.cpp · Gradio

---

## Problem Statement

Providing reliable first-aid guidance with language models presents unique challenges:

* First-aid instructions must be precise and sequential
* Hallucinated or unsafe advice can cause harm
* Internet connectivity may not always be available
* Medical-domain prompts require strong guardrails

The goal of this project is **not** to build a general medical chatbot, but to design a **constrained, retrieval-grounded assistant** that:

* Uses only verified reference material
* Produces deterministic, step-by-step instructions
* Avoids hallucination through strict prompt rules
* Runs fully offline with local models

---

## Key Features

* End-to-end offline RAG pipeline
* PDF ingestion and text normalization
* Semantic chunking with overlap for contextual continuity
* Dense embeddings using BGE-small
* FAISS-based cosine similarity search
* Local LLM inference via quantized GGUF model
* Safety-constrained prompt design
* Streaming Gradio chat interface

---

## System Architecture

```
First-Aid PDF Manuals
        |
        v
PDF Text Extraction & Cleaning
        |
        v
Semantic Chunking (Overlapping)
        |
        v
Dense Embeddings (BGE-small)
        |
        v
FAISS Vector Index (Cosine Similarity)
        |
        v
Context Assembly (Top-k Chunks)
        |
        v
Local LLM Inference (LLaMA.cpp)
        |
        v
Gradio Chat Interface
```

---

## Retrieval-Augmented Generation Design

This system enforces **retrieval-first generation**:

* The LLM is never allowed to answer from prior knowledge
* Only retrieved context is included in the prompt
* Context length is capped to prevent dilution
* Deterministic decoding (`temperature = 0.0`) is used

Prompt rules ensure:

* Numbered procedural steps
* Simple, non-technical language
* No repetition or speculation
* Emergency escalation when unconsciousness is mentioned

---

## Data

### Source

The knowledge base is built from multiple **publicly available first-aid reference manuals**.

Due to licensing and redistribution constraints, **PDF files are not included in this repository**.

To reproduce the pipeline:

1. Download reputable first-aid PDF manuals (e.g., Red Cross, IFRC, St John Ambulance)
2. Place them in `data/raw_pdfs/`
3. Run the ingestion scripts

### Data Handling

* Page numbers and layout artifacts are removed
* Excess whitespace is normalized
* Long documents are split into overlapping semantic chunks

---

## Project Structure

```
first-aid-rag-assistant/
├── src/                # Core pipeline and application code
│   ├── ingest.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── prompt.py
│   ├── rag.py
│   └── app.py
│
├── notebooks/          # Exploratory notebook
├── data/               # Ignored (PDFs and extracted text)
├── models/             # Ignored (local GGUF models)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Shreyas-Gaikwad/first-aid-rag-assistant.git
cd first-aid-rag-assistant
```

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

---

## Usage

### 1. Ingest PDFs

```bash
python -m src.ingest
```

### 2. Chunk Documents

```bash
python -m src.chunking
```

### 3. Launch the Assistant

```bash
python -m src.app
```

Open the Gradio interface at:

```
http://127.0.0.1:7860
```

---

## Safety Disclaimer

This project is intended for **educational and research purposes only**.

It does **not** replace professional medical training or emergency services.
In real emergencies, users should always contact local emergency responders.

---

## Contributing

Contributions are welcome.

Please open issues or submit pull requests for:

* Bug fixes
* Improvements to retrieval or chunking
* Prompt safety enhancements
* Documentation updates

---

## Author

Built by **Shreyas Gaikwad**
Applied Machine Learning · Retrieval-Augmented Generation · Offline AI Systems

