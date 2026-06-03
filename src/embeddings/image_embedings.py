class ImageIndexer:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.image_store = []

    def add_images(self, images, captions):
        for img, caption in zip(images, captions):
            emb = self.embedding_model.embed_query(caption)

            self.image_store.append({
                "image_path": img["path"],
                "caption": caption,
                "embedding": emb
            })

    def search(self, query):
        q_emb = self.embedding_model.embed_query(query)

        scored = []

        for item in self.image_store:
            score = self._cosine(q_emb, item["embedding"])
            scored.append((score, item))

        scored.sort(reverse=True, key=lambda x: x[0])

        return [x[1] for x in scored[:3]]

    def _cosine(self, a, b):
        import numpy as np
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))