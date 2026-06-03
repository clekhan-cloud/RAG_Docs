from src.retrieval.rag_pipeline import RAGPipeline
from src.evaluation.rag_evaluator import RAGEvaluator
from src.llm.gemini_client import GeminiClient


rag = RAGPipeline()
llm = GeminiClient()

evaluator = RAGEvaluator(llm)

query = "What is IFC's total investment in 2024?"

result = rag.ask(query)

answer = result["answer"]
contexts = result["contexts"]

print("\n--- ANSWER ---")
print(answer)

print("\n--- EVALUATION ---")

print("Context Relevance:", evaluator.evaluate_context_relevance(query, contexts))
print("Faithfulness:", evaluator.evaluate_faithfulness(query, contexts, answer))
print("Answer Relevance:", evaluator.evaluate_answer_relevance(query, answer))