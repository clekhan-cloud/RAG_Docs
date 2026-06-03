from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import uuid


class QdrantVectorStore:

    def __init__(self, embedding_model):

        self.embedding_model = embedding_model

        self.client = QdrantClient(":memory:")

        self.collection_name = "ifc_rag"

        self.dim = 384

        self._create_collection()

    # -------------------------
    # Create collection
    # -------------------------
    def _create_collection(self):

        collections = self.client.get_collections().collections

        existing = [c.name for c in collections]

        if self.collection_name not in existing:

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dim,
                    distance=Distance.COSINE
                )
            )

    # -------------------------
    # Add docs
    # -------------------------
    def add_documents(self, texts, metadatas):

        embeddings = self.embedding_model.embed_documents(texts)

        points = []

        for i, emb in enumerate(embeddings):

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=emb,
                    payload={
                        "text": texts[i],
                        "metadata": metadatas[i]
                    }
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    # -------------------------
    # Search
    # -------------------------
    def search(self, query, k=3):

        query_vector = self.embedding_model.embed_query(query)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k
        )

        docs = []

        for r in results:

            docs.append({
                "text": r.payload["text"],
                "metadata": r.payload["metadata"],
                "score": r.score
            })

        return docs

    # -------------------------
    # Save / Load placeholders
    # -------------------------
    def save_index(self):
        pass

    def load_index(self):
        pass