class ContentRouter:

    # =====================================================
    # ROUTE QUERY TYPE
    # =====================================================
    def route(self, query):

        query = query.lower()

        # -------------------------------------------------
        # TABLE QUERIES
        # -------------------------------------------------
        table_keywords = [
            "table",
            "revenue",
            "financial",
            "amount",
            "numbers",
            "statistics",
            "growth",
            "profit",
            "investment",
            "%"
        ]

        # -------------------------------------------------
        # IMAGE QUERIES
        # -------------------------------------------------
        image_keywords = [
            "chart",
            "graph",
            "image",
            "figure",
            "diagram",
            "visual",
            "trend",
            "map"
        ]

        # -------------------------------------------------
        # CHECK TABLE
        # -------------------------------------------------
        for kw in table_keywords:

            if kw in query:
                return "table"

        # -------------------------------------------------
        # CHECK IMAGE
        # -------------------------------------------------
        for kw in image_keywords:

            if kw in query:
                return "image"

        # -------------------------------------------------
        # DEFAULT
        # -------------------------------------------------
        return "text"