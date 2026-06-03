from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)

import numpy as np
import uuid


class QdrantVectorStore:

    def __init__(
        self,
        embedding_model,
        collection_name="ifc_rag"
    ):

        self.embedding_model = embedding_model

        self.collection_name = collection_name

        self.client = QdrantClient(
            path="storage/qdrant"
        )

        self.documents = []

        self._create_collection()

    # =====================================================
    # CREATE COLLECTION
    # =====================================================
    def _create_collection(self):

        collections = self.client.get_collections().collections

        existing = [
            c.name for c in collections
        ]

        if self.collection_name not in existing:

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE
                )
            )

    # =====================================================
    # ADD DOCUMENTS
    # =====================================================
    def add_documents(
        self,
        texts,
        metadatas
    ):

        embeddings = self.embedding_model.embed_documents(
            texts
        )

        points = []

        for i, emb in enumerate(embeddings):

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={
                    "text": texts[i],
                    "metadata": metadatas[i]
                }
            )

            points.append(point)

            self.documents.append({
                "text": texts[i],
                "metadata": metadatas[i]
            })

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    # =====================================================
    # SEARCH
    # =====================================================
    def search(
        self,
        query,
        k=3
    ):

        query_embedding = self.embedding_model.embed_query(
            query
        )

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k
        )

        docs = []

        for r in results:

            docs.append({
                "text": r.payload["text"],
                "metadata": r.payload["metadata"]
            })

        return docs

    # alias
    def similarity_search(
        self,
        query,
        k=3
    ):

        return self.search(
            query,
            k
        )

    # =====================================================
    # LOAD INDEX
    # =====================================================
    def load_index(self):
        pass

    # =====================================================
    # MAXSIM SEARCH
    # =====================================================
    def maxsim_search(
        self,
        query,
        k=5
    ):

        return self.search(
            query,
            k
        )