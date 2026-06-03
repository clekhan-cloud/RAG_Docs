# src/benchmark/benchmark_runner.py

from src.benchmark.benchmark_loader import (
    BenchmarkLoader
)

from src.evaluation.rag_evaluator import (
    RAGEvaluator
)


class BenchmarkRunner:

    # =====================================================
    # INIT
    # =====================================================
    def __init__(
        self,
        benchmark_file,
        rag_pipeline
    ):

        # -------------------------------------------------
        # FILE PATH
        # -------------------------------------------------
        self.file_path = benchmark_file

        # -------------------------------------------------
        # RAG PIPELINE
        # -------------------------------------------------
        self.rag_pipeline = rag_pipeline

        # -------------------------------------------------
        # LOAD BENCHMARK DATASET
        # -------------------------------------------------
        self.loader = BenchmarkLoader(
            self.file_path
        )

        self.dataset = self.loader.load()

        # -------------------------------------------------
        # EVALUATOR
        # -------------------------------------------------
        self.evaluator = RAGEvaluator()

    # =====================================================
    # RUN BENCHMARK
    # =====================================================
    def run(self):

        print("\n" + "=" * 70)
        print("🚀 RUNNING BENCHMARK")
        print("=" * 70)

        results = []

        for idx, row in self.dataset.iterrows():

            question = row.get(
                "question",
                ""
            )

            ground_truth = row.get(
                "ground_truth",
                ""
            )

            print(f"\n🔎 Question {idx + 1}")
            print(f"Q: {question}")

            try:

                # -----------------------------------------
                # RETRIEVE DOCS
                # -----------------------------------------
                docs = self.rag_pipeline.retriever.retrieve(
                    query=question,
                    k=5
                )

                # -----------------------------------------
                # GENERATE ANSWER
                # -----------------------------------------
                response = self.rag_pipeline.generate_answer(
                    question,
                    docs
                )

                # -----------------------------------------
                # EVALUATE
                # -----------------------------------------
                metrics = self.evaluator.evaluate(
                    prediction=response,
                    ground_truth=ground_truth
                )

                result = {

                    "question": question,

                    "ground_truth": ground_truth,

                    "prediction": response,

                    "metrics": metrics
                }

                results.append(result)

                print("✅ Completed")

            except Exception as e:

                print(f"❌ Failed: {e}")

        print("\n" + "=" * 70)
        print("✅ BENCHMARK COMPLETE")
        print("=" * 70)

        return results