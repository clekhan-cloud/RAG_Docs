class CitationManager:

    # =====================================================
    # SINGLE CITATION
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
            "Unknown"
        )

        content_type = meta.get(
            "content_type",
            "text"
        )

        return (
            f"Page {page} | "
            f"Chunk {chunk} | "
            f"{content_type.upper()} | "
            f"{source}"
        )

    # =====================================================
    # MULTIPLE CITATIONS
    # =====================================================
    def format_citations(
        self,
        docs
    ):

        citations = []

        for doc in docs:

            citations.append(
                self.format_citation(doc)
            )

        return citations