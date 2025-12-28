from llama_cpp import Llama
from src.vector_store import VectorStore
from src.prompt import build_prompt

llm = Llama(
    model_path="models/Phi-3-mini-4k-instruct-q4.gguf",
    n_ctx=2048,
    n_threads=24,
    n_threads_batch=24,
    n_batch=1024,
    temperature=0.0,
    repeat_penalty=1.08,
    verbose=False
)

store = VectorStore()


def answer(query, k=3):
    chunks = store.retrieve(query, k)
    prompt = build_prompt(query, chunks)

    output = llm(
        prompt,
        max_tokens=150,
        stop=["###", "<|assistant|>", "<|user|>"]
    )

    return output["choices"][0]["text"].strip()