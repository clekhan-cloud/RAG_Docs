from difflib import SequenceMatcher


class BenchmarkEvaluator:

    # =====================================================
    # TEXT SIMILARITY
    # =====================================================
    def similarity(
        self,
        text1,
        text2
    ):

        if not text1 or not text2:
            return 0.0

        return round(
            SequenceMatcher(
                None,
                text1.lower(),
                text2.lower()
            ).ratio(),
            3
        )

    # =====================================================
    # MAIN EVALUATION
    # =====================================================
    def evaluate(
        self,
        benchmark_results
    ):

        total = len(
            benchmark_results
        )

        if total == 0:

            return {

                "retrieval_accuracy": 0.0,
                "page_accuracy": 0.0,
                "answer_similarity": 0.0,
                "citation_accuracy": 0.0
            }

        retrieval_hits = 0
        page_hits = 0
        citation_hits = 0

        similarity_scores = []

        # -------------------------------------------------
        # LOOP RESULTS
        # -------------------------------------------------
        for row in benchmark_results:

            retrieved_docs = row.get(
                "retrieved_docs",
                []
            )

            expected_page = str(
                row.get(
                    "expected_page",
                    ""
                )
            )

            generated_answer = row.get(
                "generated_answer",
                ""
            )

            ground_truth_answer = row.get(
                "ground_truth_answer",
                ""
            )

            # =============================================
            # ANSWER SIMILARITY
            # =============================================
            sim = self.similarity(
                generated_answer,
                ground_truth_answer
            )

            similarity_scores.append(sim)

            # =============================================
            # RETRIEVAL ACCURACY
            # =============================================
            if len(retrieved_docs) > 0:
                retrieval_hits += 1

            # =============================================
            # PAGE ACCURACY
            # =============================================
            found_page = False

            for doc in retrieved_docs:

                metadata = doc.get(
                    "metadata",
                    {}
                )

                page = str(
                    metadata.get(
                        "page",
                        ""
                    )
                )

                if page == expected_page:

                    found_page = True
                    break

            if found_page:
                page_hits += 1

            # =============================================
            # CITATION ACCURACY
            # =============================================
            if found_page and len(retrieved_docs) > 0:
                citation_hits += 1

        # -------------------------------------------------
        # FINAL METRICS
        # -------------------------------------------------
        retrieval_accuracy = round(
            retrieval_hits / total,
            3
        )

        page_accuracy = round(
            page_hits / total,
            3
        )

        citation_accuracy = round(
            citation_hits / total,
            3
        )

        answer_similarity = round(
            sum(similarity_scores) / total,
            3
        )

        return {

            "retrieval_accuracy": retrieval_accuracy,

            "page_accuracy": page_accuracy,

            "answer_similarity": answer_similarity,

            "citation_accuracy": citation_accuracy
        }