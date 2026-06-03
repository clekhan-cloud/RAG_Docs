class GeminiClient:

    def __init__(self):
        import google.generativeai as genai

        self.genai = genai
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    # -----------------------------
    # FIXED METHOD (IMPORTANT)
    # -----------------------------
    def generate_answer(self, query: str, context: str):

        prompt = f"""
You are a financial AI assistant.

Use the context below to answer the question.

Context:
{context}

Question:
{query}

Answer clearly and accurately.
"""

        response = self.model.generate_content(prompt)

        return response.text