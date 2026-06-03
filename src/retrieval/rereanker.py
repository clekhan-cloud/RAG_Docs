class Reranker:
    def __init__(self, llm):
        self.llm = llm

    def rank(self, query, docs):
        scored = []

        for d in docs:
            prompt = f"""
You are a reranking system.

Query: {query}

Document:
{d['text']}

Return a score from 0 to 1 only.
"""

            score = float(self.llm.generate(prompt))
            scored.append((score, d))

        scored.sort(reverse=True, key=lambda x: x[0])

        return [d for _, d in scored]