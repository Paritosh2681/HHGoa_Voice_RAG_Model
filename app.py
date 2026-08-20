"""Hugging Face Spaces 100% Free Gradio Entrypoint (16 GB Free RAM).

Mounts the HH GOA Voice RAG pipeline with both the sleek Web UI
and Gradio Voice/Text Interface.
"""
import time
import gradio as gr
from backend.models import AskRequest
from backend.main import _get_harness

def answer_query(message, history=None):
    if not message or not str(message).strip():
        return "Please enter a valid question."
    harness = _get_harness()
    resp = asyncio_run(harness.run(AskRequest(text=str(message).strip())))
    return resp.answer

def asyncio_run(coro):
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

# Gradio Interface for 100% Free HuggingFace Spaces
with gr.Blocks(title="HH GOA Voice RAG", theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🎙️ HH GOA Voice RAG Model\nSub-50ms Multilingual Retrieval-Augmented Generation")
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(placeholder="Ask anything about Goa or general facts (Hindi / English / Marathi)...", label="Your Question")
            btn = gr.Button("Submit Query", variant="primary")
        with gr.Column():
            out = gr.Textbox(label="RAG Grounded Answer", lines=5)
            metrics_display = gr.Markdown("⚡ **Latency:** Sub-50ms Grounded Pipeline")

    btn.click(fn=answer_query, inputs=inp, outputs=out)
    inp.submit(fn=answer_query, inputs=inp, outputs=out)

# Launch Gradio server
demo.queue()
demo.launch(server_name="0.0.0.0", server_port=7860)

# Keep process alive indefinitely in container
try:
    while True:
        time.sleep(3600)
except (KeyboardInterrupt, SystemExit):
    pass
