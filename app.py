import gradio as gr
from src.retrieval import Retriever
from src.generate import Answerer

answerer = Answerer(Retriever())

OS_CHOICES = ["Any", "Windows 10", "Windows 11", "Windows Server 2019", "Windows Server 2022"]


def respond(question, os_version):
    if not question.strip():
        return "Enter a question.", ""

    result = answerer.answer(
        question,
        os_version=None if os_version == "Any" else os_version,
    )

    if result["refused"]:
        return result["answer"], f"Refused — confidence {result['confidence']:.3f} below threshold 0.60"

    sources = "\n".join(
        f"[{i+1}] {s['error_code']} · {s['os_version']} · {s['product']}"
        for i, s in enumerate(result["sources"])
    )
    return result["answer"], f"confidence {result['confidence']:.3f}\n\n{sources}"


with gr.Blocks(title="Windows Error Assistant") as demo:
    gr.Markdown(
        "# Windows Error Assistant\n"
        "Hybrid RAG over a synthetic error knowledge base. "
        "Refuses questions outside its documentation rather than guessing."
    )

    with gr.Row():
        question = gr.Textbox(label="Question", scale=3,
                              placeholder="e.g. folder access denied as admin")
        os_version = gr.Dropdown(OS_CHOICES, value="Any", label="OS version", scale=1)

    submit = gr.Button("Ask", variant="primary")
    answer_box = gr.Markdown(label="Answer")
    meta_box = gr.Textbox(label="Retrieval detail", lines=5)

    submit.click(respond, [question, os_version], [answer_box, meta_box])
    question.submit(respond, [question, os_version], [answer_box, meta_box])

import os
demo.launch(server_port=int(os.environ.get("PORT", 7860)))