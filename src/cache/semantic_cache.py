import numpy as np


class SemanticCache:
    def __init__(self, embedding_model, threshold=0.85):
        self.embedding_model = embedding_model
        self.threshold = threshold

        self.cache = []  # (embedding, query, answer)

    def _cosine_similarity(self, a, b):
        a = np.array(a)
        b = np.array(b)

        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get(self, query):
        query_emb = self.embedding_model.embed_query(query)

        for emb, q, ans in self.cache:
            score = self._cosine_similarity(query_emb, emb)

            if score >= self.threshold:
                return ans

        return None

    def set(self, query, answer):
        emb = self.embedding_model.embed_query(query)
        self.cache.append((emb, query, answer))