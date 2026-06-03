from src.retrieval.rag_pipeline import RAGPipeline


rag = RAGPipeline()

query = "What is IFC's mission?"

response = rag.ask(query)

print("\nQUESTION:\n")
print(response["query"])

print("\nANSWER:\n")
print(response["answer"])

print("\nRETRIEVED CONTEXT:\n")
print(response["retrieved_context"])