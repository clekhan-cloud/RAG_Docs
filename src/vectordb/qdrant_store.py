# src/vectordb/qdrant_store.py

from qdrant_client import QdrantClient

from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)

import uuid


class QdrantVectorStore:

    # =====================================================
    # INIT
    # =====================================================
    def __init__(
        self,
        embedding_model,
        collection_name="ifc_rag"
    ):

        self.embedding_model = embedding_model

        self.collection_name = collection_name

        # =================================================
        # REUSE SINGLE CLIENT
        # =================================================
        if not hasattr(QdrantVectorStore, "_shared_client"):
            QdrantVectorStore._shared_client = QdrantClient(
              path="storage/qdrant_db"
           )

        self.client = QdrantVectorStore._shared_client

        # =================================================
        # EMBEDDING DIMENSION
        # =================================================
        self.dim = 384

        # =================================================
        # LOCAL CACHE
        # =================================================
        self.documents = []

        self._create_collection()

    # =====================================================
    # CREATE COLLECTION
    # =====================================================
    def _create_collection(self):

        collections = (
            self.client.get_collections()
            .collections
        )

        existing = [
            c.name
            for c in collections
        ]

        if self.collection_name not in existing:

            self.client.create_collection(
                collection_name=self.collection_name,

                vectors_config=VectorParams(
                    size=self.dim,
                    distance=Distance.COSINE
                )
            )

            print(
                f"✅ Created collection: "
                f"{self.collection_name}"
            )

    # =====================================================
    # ADD DOCUMENTS
    # =====================================================
    def add_documents(
        self,
        texts,
        metadatas
    ):

        embeddings = (
            self.embedding_model.embed_documents(
                texts
            )
        )

        points = []

        self.documents = []

        for i, emb in enumerate(embeddings):

            metadata = metadatas[i]

            metadata.setdefault(
                "page",
                "NA"
            )

            metadata.setdefault(
                "source",
                "Unknown"
            )

            metadata.setdefault(
                "content_type",
                "text"
            )

            metadata.setdefault(
                "chunk_id",
                i
            )

            payload = {

                "text": texts[i],

                "metadata": metadata,

                "citation": (
                    f"{metadata.get('source')} | "
                    f"Page {metadata.get('page')} | "
                    f"Chunk {metadata.get('chunk_id')} | "
                    f"Type: {metadata.get('content_type')}"
                )
            }

            point = PointStruct(

                id=str(uuid.uuid4()),

                vector=emb,

                payload=payload
            )

            points.append(point)

            # local cache
            self.documents.append({

                "text": texts[i],

                "metadata": metadata,

                "citation": payload["citation"]
            })

        # =================================================
        # UPSERT
        # =================================================
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        print(
            f"✅ Added {len(points)} "
            f"documents to Qdrant"
        )

    # =====================================================
    # SIMILARITY SEARCH
    # =====================================================
    def similarity_search(
        self,
        query,
        k=5
    ):

        # =================================================
        # QUERY EMBEDDING
        # =================================================
        query_emb = (
            self.embedding_model.embed_query(
                query
            )
        )

        # =================================================
        # QDRANT QUERY
        # =================================================
        results = self.client.query_points(
            collection_name=self.collection_name,

            query=query_emb,

            limit=k
        ).points

        docs = []

        # =================================================
        # FORMAT RESULTS
        # =================================================
        for rank, r in enumerate(results):

            payload = r.payload

            metadata = payload.get(
                "metadata",
                {}
            )

            metadata.setdefault(
                "page",
                "NA"
            )

            metadata.setdefault(
                "source",
                "Unknown"
            )

            metadata.setdefault(
                "content_type",
                "text"
            )

            metadata.setdefault(
                "chunk_id",
                rank
            )

            doc = {

                "text": payload.get(
                    "text",
                    ""
                ),

                "metadata": metadata,

                "score": round(
                    float(r.score),
                    4
                ),

                "retrieval_rank": rank + 1,

                "citation": payload.get(
                    "citation",
                    (
                        f"{metadata.get('source')} | "
                        f"Page {metadata.get('page')} | "
                        f"Chunk {metadata.get('chunk_id')} | "
                        f"Type: {metadata.get('content_type')}"
                    )
                )
            }

            # =============================================
            # OPTIONAL IMAGE SUPPORT
            # =============================================
            if "image_path" in payload:

                doc["image_path"] = payload[
                    "image_path"
                ]

            if "image_caption" in payload:

                doc["image_caption"] = payload[
                    "image_caption"
                ]

            docs.append(doc)

        return docs

    # =====================================================
    # MULTIMODAL SEARCH
    # =====================================================
    def maxsim_search(
        self,
        query,
        k=5
    ):

        return self.similarity_search(
            query=query,
            k=k
        )

    # =====================================================
    # FILTER BY CONTENT TYPE
    # =====================================================
    def search_by_content_type(
        self,
        query,
        content_type="text",
        k=5
    ):

        results = self.similarity_search(
            query=query,
            k=20
        )

        filtered = []

        for doc in results:

            meta = doc.get(
                "metadata",
                {}
            )

            if meta.get(
                "content_type",
                "text"
            ) == content_type:

                filtered.append(doc)

        return filtered[:k]

    # =====================================================
    # LOAD INDEX
    # =====================================================
    def load_index(self):

        try:

            points, _ = self.client.scroll(
                collection_name=self.collection_name,

                limit=10000,

                with_payload=True,

                with_vectors=False
            )

            self.documents = []

            for p in points:

                payload = p.payload

                self.documents.append({

                    "text": payload.get(
                        "text",
                        ""
                    ),

                    "metadata": payload.get(
                        "metadata",
                        {}
                    ),

                    "citation": payload.get(
                        "citation",
                        ""
                    )
                })

            print(
                f"✅ Loaded "
                f"{len(self.documents)} "
                f"documents from Qdrant"
            )

        except Exception as e:

            print(
                f"⚠️ Failed to load Qdrant "
                f"collection: {e}"
            )

    # =====================================================
    # DOCUMENT COUNT
    # =====================================================
    def count(self):

        return len(self.documents)

    # =====================================================
    # GET ALL DOCUMENTS
    # =====================================================
    def get_all_documents(self):

        return self.documents

    # =====================================================
    # CLOSE CLIENT
    # =====================================================
    def close(self):

        try:
            self.client.close()

        except Exception as e:

            print(
                f"⚠️ Error closing Qdrant client: {e}"
            )