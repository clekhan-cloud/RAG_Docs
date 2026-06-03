from src.retrieval.retriever import Retriever
from src.llm.gemini_client import GeminiClient

# =====================================================
# OPTIONAL CONTENT ROUTER
# =====================================================
try:
    from src.routing.content_router import ContentRouter
except:
    ContentRouter = None


class RAGPipeline:

    def __init__(self, db_type="faiss"):

        # =====================================================
        # RETRIEVER
        # =====================================================
        self.retriever = Retriever(
            db_type=db_type
        )

        # =====================================================
        # LLM
        # =====================================================
        self.llm = GeminiClient()

        # =====================================================
        # DATABASE TYPE
        # =====================================================
        self.db_type = db_type

        # =====================================================
        # CONTENT ROUTER
        # =====================================================
        if ContentRouter:
            self.router = ContentRouter()
        else:
            self.router = None

    # =====================================================
    # ROUTING LOGIC
    # =====================================================
    def route_query(self, query):

        """
        Route query to correct retrieval type.

        Returns:
            text
            table
            image
        """

        # -------------------------
        # Use router if available
        # -------------------------
        if self.router:
            return self.router.route(query)

        # -------------------------
        # Simple fallback routing
        # -------------------------
        q = query.lower()

        table_keywords = [
            "table",
            "revenue",
            "financial",
            "amount",
            "investment",
            "data",
            "statistics",
            "numbers"
        ]

        image_keywords = [
            "chart",
            "graph",
            "image",
            "figure",
            "diagram",
            "visual",
            "growth trend"
        ]

        # -------------------------
        # TABLE ROUTING
        # -------------------------
        for kw in table_keywords:
            if kw in q:
                return "table"

        # -------------------------
        # IMAGE ROUTING
        # -------------------------
        for kw in image_keywords:
            if kw in q:
                return "image"

        # -------------------------
        # DEFAULT
        # -------------------------
        return "text"

    # =====================================================
    # MAIN RAG PIPELINE
    # =====================================================
    def ask(self, query, mode="text"):

        # =====================================================
        # AUTO CONTENT ROUTING
        # =====================================================
        content_type = self.route_query(query)

        # =====================================================
        # PHASE 1 — BASIC TEXT RAG
        # =====================================================
        if mode == "text":

            docs = self.retriever.retrieve(
                query=query,
                k=5,
                content_type=content_type
            )

        # =====================================================
        # PHASE 3 — HYBRID SEARCH
        # =====================================================
        elif mode == "hybrid":

            docs = self.retriever.hybrid_search(
                query=query,
                k=5
            )

        # =====================================================
        # PHASE 4 — MULTI-HOP RAG
        # =====================================================
        elif mode == "multihop":

            docs = self.retriever.multi_hop_retrieve(
                query=query,
                k=5,
                hops=2
            )

        # =====================================================
        # PHASE 5 — MULTIMODAL RAG
        # =====================================================
        elif mode == "multimodal":

            docs = self.retriever.retrieve_multimodal(
                query=query,
                k=5
            )

        # =====================================================
        # PHASE 6 — VISION RAG
        # =====================================================
        elif mode == "vision":

            docs = self.retriever.retrieve(
                query=query,
                k=5,
                content_type="image"
            )

        # =====================================================
        # TABLE-SPECIFIC RETRIEVAL
        # =====================================================
        elif mode == "table":

            docs = self.retriever.retrieve(
                query=query,
                k=5,
                content_type="table"
            )

        # =====================================================
        # IMAGE-SPECIFIC RETRIEVAL
        # =====================================================
        elif mode == "image":

            docs = self.retriever.retrieve(
                query=query,
                k=5,
                content_type="image"
            )

        # =====================================================
        # DEFAULT FALLBACK
        # =====================================================
        else:

            docs = self.retriever.retrieve(
                query=query,
                k=5,
                content_type=content_type
            )

        # =====================================================
        # BUILD CONTEXT
        # =====================================================
        context = "\n\n".join([
            d.get("text", "")
            for d in docs
        ])

        # =====================================================
        # GENERATE ANSWER
        # =====================================================
        answer = self.llm.generate_answer(
            query,
            context
        )

        # =====================================================
        # ADD CITATIONS
        # =====================================================
        citations = []

        for doc in docs:

            citation = self.retriever.format_citation(
                doc
            )

            citations.append(citation)

        # =====================================================
        # RETURN FINAL OUTPUT
        # =====================================================
        return {

            "query": query,

            "mode": mode,

            "content_type": content_type,

            "vector_db": self.db_type,

            "retrieved_docs": docs,

            "citations": citations,

            "answer": answer
        }