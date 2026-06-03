from src.ingestion.ingestion_pipeline import IngestionPipeline


PDF_PATH = "data/raw/ifc-annual-report-2024-financials.pdf"


pipeline = IngestionPipeline(PDF_PATH)

documents = pipeline.run()

print(f"\nTotal Chunks Created: {len(documents)}\n")

print(documents[0])