# src/evaluation/benchmark_evaluator.py

from difflib import SequenceMatcher
import pandas as pd


class BenchmarkEvaluator:

    def __init__(self):
        pass

    # =====================================================
    # TEXT SIMILARITY
    # =====================================================
    def text_similarity(self, generated, ground_truth):

        if not generated or not ground_truth:
            return 0.0

        generated = str(generated).lower().strip()
        ground_truth = str(ground_truth).lower().strip()

        return round(
            SequenceMatcher(
                None,
                generated,
                ground_truth
            ).ratio(),
            3
        )

    # =====================================================
    # PAGE MATCH
    # =====================================================
    def page_match(self, retrieved_docs, expected_page):

        try:
            expected_page = int(expected_page)
        except:
            return False

        for doc in retrieved_docs:

            metadata = doc.get("metadata", {})

            retrieved_page = metadata.get("page")

            try:
                retrieved_page = int(retrieved_page)
            except:
                continue

            if retrieved_page == expected_page:
                return True

        return False

    # =====================================================
    # CITATION ACCURACY
    # =====================================================
    def citation_accuracy(self, retrieved_docs):

        if not retrieved_docs:
            return 0.0

        valid = 0

        for doc in retrieved_docs:

            metadata = doc.get("metadata", {})

            has_page = "page" in metadata
            has_source = "source" in metadata

            if has_page and has_source:
                valid += 1

        return round(valid / len(retrieved_docs), 3)

    # =====================================================
    # RETRIEVAL ACCURACY
    # =====================================================
    def retrieval_accuracy(
        self,
        retrieved_docs,
        ground_truth_context
    ):

        if not retrieved_docs:
            return 0.0

        gt = str(ground_truth_context).lower()

        best_score = 0.0

        for doc in retrieved_docs:

            text = doc.get("text", "").lower()

            score = SequenceMatcher(
                None,
                text,
                gt
            ).ratio()

            best_score = max(best_score, score)

        return round(best_score, 3)

    # =====================================================
    # EVALUATE SINGLE QUESTION
    # =====================================================
    def evaluate_question(
        self,
        question,
        generated_answer,
        ground_truth_answer,
        retrieved_docs,
        expected_page,
        ground_truth_context
    ):

        answer_similarity = self.text_similarity(
            generated_answer,
            ground_truth_answer
        )

        retrieval_score = self.retrieval_accuracy(
            retrieved_docs,
            ground_truth_context
        )

        page_correct = self.page_match(
            retrieved_docs,
            expected_page
        )

        citation_score = self.citation_accuracy(
            retrieved_docs
        )

        final_score = round(
            (
                answer_similarity +
                retrieval_score +
                citation_score +
                (1.0 if page_correct else 0.0)
            ) / 4,
            3
        )

        return {
            "question": question,
            "answer_similarity": answer_similarity,
            "retrieval_accuracy": retrieval_score,
            "page_accuracy": page_correct,
            "citation_accuracy": citation_score,
            "final_score": final_score
        }

    # =====================================================
    # EVALUATE FULL BENCHMARK
    # =====================================================
    def evaluate_benchmark(self, benchmark_results):

        evaluations = []

        for row in benchmark_results:

            result = self.evaluate_question(
                question=row["question"],
                generated_answer=row["generated_answer"],
                ground_truth_answer=row["ground_truth_answer"],
                retrieved_docs=row["retrieved_docs"],
                expected_page=row["expected_page"],
                ground_truth_context=row["ground_truth_context"]
            )

            evaluations.append(result)

        return evaluations

    # =====================================================
    # SUMMARY METRICS
    # =====================================================
    def summarize_results(self, evaluations):

        if not evaluations:
            return {}

        df = pd.DataFrame(evaluations)

        summary = {
            "avg_answer_similarity":
                round(df["answer_similarity"].mean(), 3),

            "avg_retrieval_accuracy":
                round(df["retrieval_accuracy"].mean(), 3),

            "avg_citation_accuracy":
                round(df["citation_accuracy"].mean(), 3),

            "page_accuracy_rate":
                round(df["page_accuracy"].mean(), 3),

            "overall_score":
                round(df["final_score"].mean(), 3)
        }

        return summary