import pandas as pd
from pathlib import Path


class BenchmarkLoader:

    def __init__(
        self,
        file_path="benchmark/RAG_evaluation_dataset(1).xlsx"
    ):

        self.file_path = file_path

    def load(self):

        if not Path(self.file_path).exists():

            print(
                f"⚠️ Benchmark file not found: "
                f"{self.file_path}"
            )

            return pd.DataFrame()

        df = pd.read_csv(
            self.file_path
        )

        print(
            f"✅ Loaded benchmark dataset: "
            f"{len(df)} rows"
        )

        return df