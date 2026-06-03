from src.retrieval.retriever import Retriever


retriever = Retriever()


query = "What is IFC mission?"


results = retriever.retrieve(
    query,
    top_k=3
)


print("\nRETRIEVED CHUNKS:\n")


for idx, result in enumerate(results):

    print(f"\nRESULT {idx+1}\n")

    print(result["chunk"])

    print("\n-------------------\n")