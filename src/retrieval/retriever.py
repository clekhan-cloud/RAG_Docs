# =====================================================
# src/retrieval/retriever.py
# =====================================================

from rank_bm25 import BM25Okapi
import numpy as np

from src.embeddings.embedding_generator import (
    EmbeddingGenerator
)

from src.vectordb.faiss_store import (
    FAISSVectorStore
)

from src.vectordb.qdrant_store import (
    QdrantVectorStore
)


class Retriever:

    # =====================================================
    # INIT
    # =====================================================
    def __init__(self, db_type="faiss"):

        # -------------------------------------------------
        # Embedding Model
        # -------------------------------------------------
        self.embedding_model = EmbeddingGenerator()

        # -------------------------------------------------
        # VECTOR DB SELECTION
        # -------------------------------------------------
        if db_type == "faiss":

            self.vector_store = FAISSVectorStore(
                self.embedding_model.model
            )

        elif db_type == "qdrant":

            self.vector_store = QdrantVectorStore(
                self.embedding_model.model
            )

        else:

            raise ValueError(
                f"Unsupported DB type: {db_type}"
            )

        # -------------------------------------------------
        # LOAD VECTOR INDEX
        # -------------------------------------------------
        self.vector_store.load_index()

        # -------------------------------------------------
        # BM25
        # -------------------------------------------------
        self.bm25 = None
        self.corpus = []

    # =====================================================
    # FILTER BY CONTENT TYPE
    # =====================================================
    def filter_by_content_type(
        self,
        docs,
        content_type=None
    ):

        if content_type is None:
            return docs

        filtered = []

        for doc in docs:

            metadata = doc.get(
                "metadata",
                {}
            )

            doc_type = metadata.get(
                "content_type",
                "text"
            )

            if doc_type == content_type:

                filtered.append(doc)

        return filtered

    # =====================================================
    # GENERIC RETRIEVAL
    # =====================================================
    def retrieve(
        self,
        query,
        k=5,
        content_type=None
    ):

        docs = self.vector_store.similarity_search(
            query=query,
            k=15
        )

        # -------------------------------------------------
        # FILTER CONTENT TYPE
        # -------------------------------------------------
        docs = self.filter_by_content_type(
            docs,
            content_type=content_type
        )

        # -------------------------------------------------
        # RERANK
        # -------------------------------------------------
        docs = self.rerank(
            docs,
            query
        )

        return docs[:k]

    # =====================================================
    # TEXT RETRIEVAL
    # =====================================================
    def retrieve_text(
        self,
        query,
        k=5
    ):

        return self.retrieve(
            query=query,
            k=k,
            content_type="text"
        )

    # =====================================================
    # TABLE RETRIEVAL
    # =====================================================
    def retrieve_tables(
        self,
        query,
        k=5
    ):

        docs = self.retrieve(
            query=query,
            k=20
        )

        table_docs = [

            d for d in docs

            if d.get(
                "metadata",
                {}
            ).get(
                "content_type"
            ) == "table"
        ]

        return table_docs[:k]

    # =====================================================
    # IMAGE RETRIEVAL
    # =====================================================
    def retrieve_images(
        self,
        query,
        k=5
    ):

        docs = self.retrieve(
            query=query,
            k=20
        )

        image_docs = [

            d for d in docs

            if d.get(
                "metadata",
                {}
            ).get(
                "content_type"
            ) == "image"
        ]

        return image_docs[:k]

    # =====================================================
    # MULTIMODAL RETRIEVAL
    # =====================================================
    def retrieve_multimodal(
        self,
        query,
        k=10
    ):

        return self.retrieve(
            query=query,
            k=k
        )

    # =====================================================
    # HYBRID + MULTIMODAL RETRIEVAL
    # =====================================================
    def hybrid_multimodal_search(
        self,
        query,
        k=10
    ):

        # -------------------------------------------------
        # DENSE SEARCH
        # -------------------------------------------------
        dense_docs = (
            self.vector_store.similarity_search(
                query=query,
                k=10
            )
        )

        # -------------------------------------------------
        # MULTIMODAL SEARCH
        # -------------------------------------------------
        multimodal_docs = (
            self.vector_store.maxsim_search(
                query=query,
                k=10
            )
        )

        # -------------------------------------------------
        # BM25 SEARCH
        # -------------------------------------------------
        if self.bm25 is None:

            self.build_bm25(
                self.vector_store.documents
            )

        bm25_scores = self.bm25.get_scores(
            query.lower().split()
        )

        top_idx = np.argsort(
            bm25_scores
        )[::-1][:10]

        bm25_docs = [

            self.vector_store.documents[i]

            for i in top_idx
        ]

        # -------------------------------------------------
        # COMBINE RESULTS
        # -------------------------------------------------
        combined = (

            dense_docs +

            multimodal_docs +

            bm25_docs
        )

        # -------------------------------------------------
        # REMOVE DUPLICATES
        # -------------------------------------------------
        seen = set()

        final_docs = []

        for doc in combined:

            key = doc.get(
                "text",
                ""
            )

            if key not in seen:

                final_docs.append(doc)

                seen.add(key)

        # -------------------------------------------------
        # FINAL RERANK
        # -------------------------------------------------
        final_docs = self.rerank(
            final_docs,
            query
        )

        return final_docs[:k]

    # =====================================================
    # PHASE 3 — RERANKING
    # =====================================================
    def rerank(
        self,
        docs,
        query
    ):

        query_words = set(
            query.lower().split()
        )

        scored = []

        for doc in docs:

            text = doc.get(
                "text",
                ""
            ).lower()

            # ---------------------------------------------
            # Lexical Overlap
            # ---------------------------------------------
            overlap_score = sum(

                1 for w in query_words

                if w in text
            )

            # ---------------------------------------------
            # Metadata
            # ---------------------------------------------
            metadata = doc.get(
                "metadata",
                {}
            )

            content_type = metadata.get(
                "content_type",
                "text"
            )

            # ---------------------------------------------
            # Bonuses
            # ---------------------------------------------
            bonus = 0.0

            if content_type == "table":

                bonus = 0.3

            elif content_type == "image":

                bonus = 0.2

            # ---------------------------------------------
            # Length Penalty
            # ---------------------------------------------
            length_penalty = min(
                len(text) / 1000,
                1.0
            )

            final_score = (

                overlap_score +

                bonus -

                (0.2 * length_penalty)
            )

            # ---------------------------------------------
            # Store Score
            # ---------------------------------------------
            doc["score"] = round(
                final_score,
                4
            )

            scored.append(
                (final_score, doc)
            )

        scored.sort(

            key=lambda x: x[0],

            reverse=True
        )

        return [

            d for _, d in scored
        ]

    # =====================================================
    # BUILD BM25
    # =====================================================
    def build_bm25(
        self,
        docs
    ):

        self.corpus = [

            d["text"]

            for d in docs
        ]

        tokenized_corpus = [

            doc.lower().split()

            for doc in self.corpus
        ]

        self.bm25 = BM25Okapi(
            tokenized_corpus
        )

    # =====================================================
    # HYBRID SEARCH
    # =====================================================
    def hybrid_search(
        self,
        query,
        k=10
    ):

        # -------------------------------------------------
        # DENSE SEARCH
        # -------------------------------------------------
        dense_docs = (
            self.vector_store.similarity_search(
                query=query,
                k=10
            )
        )

        # -------------------------------------------------
        # BM25 SEARCH
        # -------------------------------------------------
        if self.bm25 is None:

            self.build_bm25(
                self.vector_store.documents
            )

        bm25_scores = self.bm25.get_scores(
            query.lower().split()
        )

        top_idx = np.argsort(
            bm25_scores
        )[::-1][:10]

        bm25_docs = [

            self.vector_store.documents[i]

            for i in top_idx
        ]

        # -------------------------------------------------
        # COMBINE
        # -------------------------------------------------
        combined = dense_docs + bm25_docs

        # -------------------------------------------------
        # REMOVE DUPLICATES
        # -------------------------------------------------
        seen = set()

        final_docs = []

        for doc in combined:

            key = doc.get(
                "text",
                ""
            )

            if key not in seen:

                final_docs.append(doc)

                seen.add(key)

        final_docs = self.rerank(
            final_docs,
            query
        )

        return final_docs[:k]

    # =====================================================
    # MULTI-HOP RETRIEVAL
    # =====================================================
    def multi_hop_retrieve(
        self,
        query,
        k=5,
        hops=2
    ):

        current_query = query

        all_docs = []

        for _ in range(hops):

            docs = self.hybrid_multimodal_search(
                current_query,
                k=5
            )

            all_docs.extend(docs)

            if docs:

                current_query = docs[0][
                    "text"
                ][:200]

        # -------------------------------------------------
        # REMOVE DUPLICATES
        # -------------------------------------------------
        seen = set()

        final_docs = []

        for doc in all_docs:

            key = doc.get(
                "text",
                ""
            )

            if key not in seen:

                final_docs.append(doc)

                seen.add(key)

        return final_docs[:k]

    # =====================================================
    # VISION RETRIEVAL
    # =====================================================
    def vision_retrieve(
        self,
        query,
        vision_store,
        k=5
    ):

        query_emb = (
            self.embedding_model.model.embed_query(
                query
            )
        )

        return vision_store.search(
            query_emb,
            k=k
        )

    # =====================================================
    # CITATION FORMATTER
    # =====================================================
    def format_citation(
        self,
        doc
    ):

        meta = doc.get(
            "metadata",
            {}
        )

        page = meta.get(
            "page",
            "NA"
        )

        chunk = meta.get(
            "chunk_id",
            "NA"
        )

        source = meta.get(
            "source",
            "NA"
        )

        content_type = meta.get(
            "content_type",
            "text"
        )

        return (

            f"{source} | "

            f"Page {page} | "

            f"Chunk {chunk} | "

            f"Type: {content_type}"
        )

    # =====================================================
    # RETRIEVAL EXPLAINABILITY
    # =====================================================
    def explain_retrieval(
        self,
        query,
        doc
    ):

        text = doc.get(
            "text",
            ""
        ).lower()

        query_words = set(
            query.lower().split()
        )

        matches = [

            w for w in query_words

            if w in text
        ]

        return {

            "matched_terms": matches,

            "match_score": len(matches),

            "content_type": doc.get(
                "metadata",
                {}
            ).get(
                "content_type",
                "text"
            )
        }