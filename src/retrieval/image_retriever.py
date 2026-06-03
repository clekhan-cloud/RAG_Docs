class ImageRetriever:

    def __init__(self, retriever):

        self.retriever = retriever

    # =====================================================
    # IMAGE RETRIEVAL
    # =====================================================
    def retrieve_images(
        self,
        query,
        k=5
    ):

        docs = self.retriever.hybrid_search(
            query,
            k=20
        )

        image_docs = []

        for doc in docs:

            meta = doc.get(
                "metadata",
                {}
            )

            if meta.get(
                "content_type"
            ) == "image":

                image_docs.append(doc)

        return image_docs[:k]