import gradio as gr
from src.rag import answer


def chat_stream(message, history):
    history = history or []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer(message)})
    return history


with gr.Blocks(title="First Aid Assistant") as demo:
    gr.Markdown("## First Aid Assistant (Offline RAG)")
    gr.Markdown("This tool does not replace professional medical care.")

    chatbot = gr.Chatbot(height=420)
    msg = gr.Textbox(placeholder="Ask a first-aid question…", show_label=False)

    msg.submit(chat_stream, [msg, chatbot], chatbot)
    msg.submit(lambda: "", None, msg)

demo.launch()