import faiss
import numpy as np
import pickle
from pathlib import Path

from src.utils.logger import logger


class FAISSIndexer:
    def __init__(self):

        self.index = None
        self.documents = []

    def create_index(self, embeddings, documents):
        """
        Create FAISS index.
        """

        dimension = embeddings.shape[1]

        logger.info(f"Creating FAISS index with dimension {dimension}")

        self.index = faiss.IndexFlatL2(dimension)

        self.index.add(np.array(embeddings).astype("float32"))

        self.documents = documents

        logger.info(f"Indexed {len(documents)} documents")

    def save_index(self):

        Path("data/embeddings").mkdir(
            parents=True,
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            "data/embeddings/faiss_index.index"
        )

        with open(
            "data/embeddings/faiss_documents.pkl",
            "wb"
        ) as f:

            pickle.dump(self.documents, f)

        logger.info("FAISS index saved")

    def load_index(self):

        self.index = faiss.read_index(
            "data/embeddings/faiss_index.index"
        )

        with open(
            "data/embeddings/faiss_documents.pkl",
            "rb"
        ) as f:

            self.documents = pickle.load(f)

        logger.info("FAISS index loaded")

    def search(
        self,
        query_embedding,
        top_k=5
    ):
        """
        Perform similarity search.
        """

        distances, indices = self.index.search(
            np.array([query_embedding]).astype("float32"),
            top_k
        )

        results = []

        for idx in indices[0]:

            results.append(self.documents[idx])

        return results