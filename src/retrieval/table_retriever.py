class TableRetriever:

    def __init__(self, retriever):

        self.retriever = retriever

    # =====================================================
    # TABLE RETRIEVAL
    # =====================================================
    def retrieve_tables(
        self,
        query,
        k=5
    ):

        docs = self.retriever.hybrid_search(
            query,
            k=20
        )

        table_docs = []

        for doc in docs:

            meta = doc.get(
                "metadata",
                {}
            )

            if meta.get(
                "content_type"
            ) == "table":

                table_docs.append(doc)

        return table_docs[:k]