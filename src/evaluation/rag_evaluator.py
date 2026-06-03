# src/evaluation/rag_evaluator.py

from difflib import SequenceMatcher


class RAGEvaluator:

    # =====================================================
    # INIT
    # =====================================================
    def __init__(self):
        pass

    # =====================================================
    # BASIC PHASE-2 EVALUATION
    # =====================================================
    def evaluate(
        self,
        query,
        docs,
        answer
    ):

        # -------------------------------------------------
        # Retrieval Score
        # -------------------------------------------------
        retrieval_score = min(
            len(docs) / 5,
            1.0
        )

        # -------------------------------------------------
        # Answer Quality
        # -------------------------------------------------
        answer_score = (
            1.0
            if answer and len(answer) > 50
            else 0.5
        )

        # -------------------------------------------------
        # Query-Context Overlap
        # -------------------------------------------------
        query_words = set(
            query.lower().split()
        )

        overlap = 0

        for d in docs:

            text = d.get(
                "text",
                ""
            ).lower()

            overlap += sum(
                1
                for w in query_words
                if w in text
            )

        overlap_score = min(
            overlap / 10,
            1.0
        )

        # -------------------------------------------------
        # Final Score
        # -------------------------------------------------
        final_score = (
            retrieval_score +
            answer_score +
            overlap_score
        ) / 3

        return {

            "retrieval_score": round(
                retrieval_score,
                3
            ),

            "answer_score": round(
                answer_score,
                3
            ),

            "overlap_score": round(
                overlap_score,
                3
            ),

            "final_score": round(
                final_score,
                3
            ),

            "num_docs": len(docs),

            "answer_length": len(answer)
        }

    # =====================================================
    # ANSWER SIMILARITY
    # =====================================================
    def compute_similarity(
        self,
        generated_answer,
        ground_truth_answer
    ):

        if not generated_answer or not ground_truth_answer:
            return 0.0

        similarity = SequenceMatcher(
            None,
            generated_answer.lower(),
            ground_truth_answer.lower()
        ).ratio()

        return round(similarity, 3)

    # =====================================================
    # PAGE ACCURACY
    # =====================================================
    def evaluate_page_accuracy(
        self,
        retrieved_docs,
        expected_page
    ):

        retrieved_pages = []

        for doc in retrieved_docs:

            page = doc.get(
                "metadata",
                {}
            ).get(
                "page"
            )

            if page is not None:
                retrieved_pages.append(page)

        hit = expected_page in retrieved_pages

        return {

            "expected_page": expected_page,

            "retrieved_pages": retrieved_pages,

            "page_hit": hit
        }

    # =====================================================
    # CITATION ACCURACY
    # =====================================================
    def evaluate_citations(
        self,
        retrieved_docs
    ):

        citations = []

        valid = 0

        for doc in retrieved_docs:

            metadata = doc.get(
                "metadata",
                {}
            )

            page = metadata.get("page")
            source = metadata.get("source")

            citation = {
                "page": page,
                "source": source
            }

            citations.append(citation)

            if page is not None and source:
                valid += 1

        citation_score = (
            valid / len(retrieved_docs)
            if retrieved_docs
            else 0
        )

        return {

            "citation_score": round(
                citation_score,
                3
            ),

            "citations": citations
        }

    # =====================================================
    # GROUND TRUTH EVALUATION
    # =====================================================
    def evaluate_against_ground_truth(
        self,
        query,
        retrieved_docs,
        generated_answer,
        ground_truth_answer,
        expected_page
    ):

        # -------------------------------------------------
        # Basic Retrieval Evaluation
        # -------------------------------------------------
        basic_eval = self.evaluate(
            query=query,
            docs=retrieved_docs,
            answer=generated_answer
        )

        # -------------------------------------------------
        # Answer Similarity
        # -------------------------------------------------
        similarity_score = self.compute_similarity(
            generated_answer,
            ground_truth_answer
        )

        # -------------------------------------------------
        # Page Accuracy
        # -------------------------------------------------
        page_eval = self.evaluate_page_accuracy(
            retrieved_docs,
            expected_page
        )

        # -------------------------------------------------
        # Citation Accuracy
        # -------------------------------------------------
        citation_eval = self.evaluate_citations(
            retrieved_docs
        )

        # -------------------------------------------------
        # Overall Benchmark Score
        # -------------------------------------------------
        benchmark_score = (

            basic_eval["final_score"] +

            similarity_score +

            citation_eval["citation_score"]

        ) / 3

        # -------------------------------------------------
        # Final Output
        # -------------------------------------------------
        return {

            # ---------------------------------------------
            # Basic Evaluation
            # ---------------------------------------------
            "retrieval_score":
                basic_eval["retrieval_score"],

            "answer_score":
                basic_eval["answer_score"],

            "overlap_score":
                basic_eval["overlap_score"],

            "final_score":
                basic_eval["final_score"],

            # ---------------------------------------------
            # Ground Truth Comparison
            # ---------------------------------------------
            "answer_similarity":
                similarity_score,

            "ground_truth_answer":
                ground_truth_answer,

            "generated_answer":
                generated_answer,

            # ---------------------------------------------
            # Page Evaluation
            # ---------------------------------------------
            "expected_page":
                page_eval["expected_page"],

            "retrieved_pages":
                page_eval["retrieved_pages"],

            "page_hit":
                page_eval["page_hit"],

            # ---------------------------------------------
            # Citation Evaluation
            # ---------------------------------------------
            "citation_score":
                citation_eval["citation_score"],

            "citations":
                citation_eval["citations"],

            # ---------------------------------------------
            # Overall Benchmark
            # ---------------------------------------------
            "benchmark_score":
                round(benchmark_score, 3)
        }