import faiss
import numpy as np


class VisionFAISSStore:

    def __init__(self, dim=512):
        self.index = faiss.IndexFlatL2(dim)
        self.patches = []

    def add(self, embeddings, metadata):
        embeddings = np.array(embeddings).astype("float32")
        self.index.add(embeddings)

        self.patches.extend(metadata)

    def search(self, query_embedding, k=3):

        query_embedding = np.array([query_embedding]).astype("float32")

        distances, indices = self.index.search(query_embedding, k)

        results = []
        for idx in indices[0]:
            if idx < len(self.patches):
                results.append(self.patches[idx])

        return results