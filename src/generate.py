import os
from groq import Groq
from src import config
from src.guardrails import check_confidence

PROMPT = """You are a Windows troubleshooting assistant.

Answer using ONLY the documentation below. Do not add fixes from your own
knowledge. Cite the source number you used, like [1]. If the documentation
does not cover the question, say so plainly instead of guessing.

DOCUMENTATION:
{context}

QUESTION: {question}"""


class Answerer:
    def __init__(self, retriever):
        self.retriever = retriever
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])

    def answer(self, question, os_version=None, k=config.TOP_K):
        allowed, score = check_confidence(self.retriever, question)
        if not allowed:
            return {
                "answer": "I don't have documentation covering that topic.",
                "sources": [],
                "confidence": score,
                "refused": True,
            }

        hits = self.retriever.search(question, os_version=os_version, k=k)
        if not hits:
            return {
                "answer": f"No documentation found for {os_version}.",
                "sources": [],
                "confidence": score,
                "refused": True,
            }

        context = "\n\n".join(
            f"[{i+1}] {c['text']}" for i, (c, _) in enumerate(hits)
        )
        resp = self.client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user",
                       "content": PROMPT.format(context=context, question=question)}],
            temperature=0.2,
        )
        return {
            "answer": resp.choices[0].message.content,
            "sources": [c for c, _ in hits],
            "confidence": score,
            "refused": False,
        }


if __name__ == "__main__":
    from src.retrieval import Retriever

    a = Answerer(Retriever())

    for q in [f"how do I fix error {a.retriever.chunks[0]['error_code']}",
              "how do I bake sourdough bread"]:
        result = a.answer(q)
        print(f"\nQ: {q}")
        print(f"confidence {result['confidence']:.3f}  refused={result['refused']}")
        print(result["answer"][:300])