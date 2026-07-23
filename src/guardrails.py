from src import config


def check_confidence(retriever, question, threshold=config.CONFIDENCE_THRESHOLD):
    """Decide whether the knowledge base plausibly covers this question.

    Retrieval always returns something — there is no null result — so without
    an explicit floor, an off-topic question still yields three confident-looking
    documents. Threshold set at 0.60, the midpoint of a measured 0.167 gap
    between on-topic (min 0.692) and off-topic (max 0.524) queries across a
    16-query validation set.
    """
    score = retriever.best_score(question)
    return score >= threshold, score