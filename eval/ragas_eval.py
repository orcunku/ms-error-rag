import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import FastEmbedEmbeddings

from src.retrieval import Retriever
from src.generate import Answerer
from src import config

QUESTIONS = [
    "how do I fix a folder access denied error as administrator",
    "my printer jobs are stuck in the queue",
    "windows update keeps failing partway through",
    "system shows a stop screen on startup",
    "an application crashes when I open a document",
    "network connection drops intermittently",
    "windows says the system is not activated",
    "file operations fail with an I/O error",
]


def build_dataset():
    retriever = Retriever()
    answerer = Answerer(retriever)

    rows = []
    for q in QUESTIONS:
        result = answerer.answer(q)
        if result["refused"]:
            print(f"skipped (refused): {q}")
            continue
        rows.append({
            "question": q,
            "answer": result["answer"],
            "contexts": [s["text"] for s in result["sources"]],
        })
        print(f"answered: {q}")

    return Dataset.from_list(rows)


if __name__ == "__main__":
    dataset = build_dataset()
    print(f"\nevaluating {len(dataset)} answers...\n")

    judge = LangchainLLMWrapper(ChatGroq(
        model=config.LLM_MODEL,
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0,
    ))
    embedder = LangchainEmbeddingsWrapper(
        FastEmbedEmbeddings(model_name=config.EMBED_MODEL)
    )

    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=judge,
        embeddings=embedder,
    )
    print(scores)