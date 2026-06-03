# src/vectordb/faiss_store.py

import os
import pickle
import faiss
import numpy as np


class FAISSVectorStore:

    # =====================================================
    # INIT
    # =====================================================
    def __init__(
        self,
        embedding_model,
        index_path="storage/faiss.index",
        metadata_path="storage/faiss_metadata.pkl"
    ):

        self.embedding_model = embedding_model

        self.index_path = index_path
        self.metadata_path = metadata_path

        self.index = None

        # IMPORTANT
        self.documents = []

    # =====================================================
    # CREATE INDEX
    # =====================================================
    def create_index(self, documents):

        self.documents = documents

        # -------------------------------------------------
        # EXTRACT TEXTS
        # -------------------------------------------------
        texts = [
            d.get("text", "")
            for d in documents
        ]

        # -------------------------------------------------
        # GENERATE EMBEDDINGS
        # -------------------------------------------------
        embeddings = self.embedding_model.embed_documents(
            texts
        )

        embeddings = np.array(
            embeddings,
            dtype=np.float32
        )

        # -------------------------------------------------
        # CREATE FAISS INDEX
        # -------------------------------------------------
        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(
            dimension
        )

        self.index.add(embeddings)

        # -------------------------------------------------
        # CREATE STORAGE DIRECTORY
        # -------------------------------------------------
        os.makedirs(
            "storage",
            exist_ok=True
        )

        # -------------------------------------------------
        # SAVE FAISS INDEX
        # -------------------------------------------------
        faiss.write_index(
            self.index,
            self.index_path
        )

        # -------------------------------------------------
        # SAVE DOCUMENTS + METADATA
        # -------------------------------------------------
        with open(
            self.metadata_path,
            "wb"
        ) as f:

            pickle.dump(
                self.documents,
                f
            )

        print(
            f"✅ FAISS index saved with "
            f"{len(documents)} documents"
        )

    # =====================================================
    # LOAD INDEX
    # =====================================================
    def load_index(self):

        if not os.path.exists(
            self.index_path
        ):
            print("⚠️ FAISS index not found")
            return

        # -------------------------------------------------
        # LOAD FAISS INDEX
        # -------------------------------------------------
        self.index = faiss.read_index(
            self.index_path
        )

        # -------------------------------------------------
        # LOAD DOCUMENTS
        # -------------------------------------------------
        with open(
            self.metadata_path,
            "rb"
        ) as f:

            self.documents = pickle.load(f)

        print(
            f"✅ FAISS index loaded with "
            f"{len(self.documents)} documents"
        )

    # =====================================================
    # SIMILARITY SEARCH
    # =====================================================
    def similarity_search(
        self,
        query,
        k=5
    ):

        # -------------------------------------------------
        # SAFETY CHECK
        # -------------------------------------------------
        if self.index is None:
            raise ValueError(
                "FAISS index is not loaded. "
                "Run ingestion pipeline first."
            )

        # -------------------------------------------------
        # QUERY EMBEDDING
        # -------------------------------------------------
        query_embedding = (
            self.embedding_model.embed_query(
                query
            )
        )

        query_embedding = np.array(
            [query_embedding],
            dtype=np.float32
        )

        # -------------------------------------------------
        # SEARCH
        # -------------------------------------------------
        distances, indices = self.index.search(
            query_embedding,
            k
        )

        results = []

        # -------------------------------------------------
        # FORMAT RESULTS
        # -------------------------------------------------
        for rank, idx in enumerate(indices[0]):

            if idx < len(self.documents):

                doc = self.documents[idx]

                # -----------------------------------------
                # ENSURE METADATA EXISTS
                # -----------------------------------------
                metadata = doc.get(
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
                    idx
                )

                # -----------------------------------------
                # ADD RETRIEVAL SCORE
                # -----------------------------------------
                distance = float(
                    distances[0][rank]
                )

                similarity_score = round(
                    1 / (1 + distance),
                    4
                )

                # -----------------------------------------
                # BUILD RESULT
                # -----------------------------------------
                result = {

                    "text": doc.get(
                        "text",
                        ""
                    ),

                    "metadata": metadata,

                    "score": similarity_score,

                    "retrieval_rank": rank + 1,

                    "citation": (
                        f"{metadata.get('source')} | "
                        f"Page {metadata.get('page')} | "
                        f"Chunk {metadata.get('chunk_id')} | "
                        f"Type: {metadata.get('content_type')}"
                    )
                }

                # -----------------------------------------
                # OPTIONAL IMAGE FIELDS
                # -----------------------------------------
                if "image_path" in doc:
                    result["image_path"] = doc[
                        "image_path"
                    ]

                if "image_caption" in doc:
                    result["image_caption"] = doc[
                        "image_caption"
                    ]

                results.append(result)

        return results

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
    # GET DOCUMENT COUNT
    # =====================================================
    def count(self):

        return len(self.documents)

    # =====================================================
    # GET ALL DOCUMENTS
    # =====================================================
    def get_all_documents(self):

        return self.documents