# src/llm/gemini_client.py

import os
import PIL.Image

import google.generativeai as genai


class GeminiClient:

    # =====================================================
    # INIT
    # =====================================================
    def __init__(self):

        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "❌ GEMINI_API_KEY not found "
                "in environment variables"
            )

        genai.configure(
            api_key=api_key
        )

        # -------------------------------------------------
        # Gemini Model
        # -------------------------------------------------
        self.model = genai.GenerativeModel(
            "gemini-2.0-flash"
        )

    # =====================================================
    # BASIC TEXT GENERATION
    # =====================================================
    def generate_answer(
        self,
        query,
        context
    ):

        prompt = f"""
You are an IFC Annual Report assistant.

Use ONLY the provided context.

If the answer is not available,
say:
"I could not find the answer in the document."

====================================================

CONTEXT:
{context}

====================================================

QUESTION:
{query}

====================================================

Instructions:
- Answer clearly
- Answer concisely
- Use factual information only
- Mention tables/charts if relevant
- Mention page-level evidence if present

FINAL ANSWER:
"""

        response = self.model.generate_content(
            prompt
        )

        return response.text

    # =====================================================
    # TABLE REASONING
    # =====================================================
    def reason_over_table(
        self,
        query,
        table_text
    ):

        prompt = f"""
You are a financial table analysis assistant.

You are given a table extracted
from IFC Annual Report 2024.

====================================================

TABLE:
{table_text}

====================================================

QUESTION:
{query}

====================================================

Instructions:
- Analyze the table carefully
- Extract numerical insights
- Compare values if needed
- Mention trends if present
- Give a concise answer

FINAL ANSWER:
"""

        response = self.model.generate_content(
            prompt
        )

        return response.text

    # =====================================================
    # IMAGE CAPTIONING
    # =====================================================
    def caption_image(
        self,
        image_path
    ):

        try:

            image = PIL.Image.open(
                image_path
            )

            response = self.model.generate_content([
                """
Describe this IFC report image for retrieval.

Focus on:
- chart type
- visual trends
- important labels
- IFC-related topics
- financial/environmental indicators

Keep it concise but descriptive.
""",
                image
            ])

            return response.text

        except Exception as e:

            return (
                f"Image caption generation failed: {e}"
            )

    # =====================================================
    # IMAGE QUESTION ANSWERING
    # =====================================================
    def reason_over_image(
        self,
        query,
        image_path,
        image_caption=None
    ):

        try:

            image = PIL.Image.open(
                image_path
            )

            prompt = f"""
You are an IFC visual reasoning assistant.

====================================================

IMAGE CAPTION:
{image_caption}

====================================================

QUESTION:
{query}

====================================================

Instructions:
- Analyze the image carefully
- Use chart/table/diagram evidence
- Mention trends if visible
- Use IFC financial context if relevant

FINAL ANSWER:
"""

            response = self.model.generate_content([
                prompt,
                image
            ])

            return response.text

        except Exception as e:

            return (
                f"Image reasoning failed: {e}"
            )

    # =====================================================
    # MULTIMODAL REASONING
    # =====================================================
    def multimodal_reasoning(
        self,
        query,
        docs
    ):

        context_parts = []

        for doc in docs:

            text = doc.get(
                "text",
                ""
            )

            metadata = doc.get(
                "metadata",
                {}
            )

            content_type = metadata.get(
                "content_type",
                "text"
            )

            page = metadata.get(
                "page",
                "NA"
            )

            context_parts.append(
                f"""
[TYPE: {content_type}]
[PAGE: {page}]

{text}
"""
            )

        combined_context = "\n\n".join(
            context_parts
        )

        prompt = f"""
You are a multimodal IFC RAG assistant.

You may receive:
- text chunks
- financial tables
- image captions
- charts/visual summaries

====================================================

MULTIMODAL CONTEXT:
{combined_context}

====================================================

QUESTION:
{query}

====================================================

Instructions:
- Combine evidence across modalities
- Use table/image insights if relevant
- Mention trends/numbers accurately
- Ground answers in retrieved context only
- Mention page evidence when possible

FINAL ANSWER:
"""

        response = self.model.generate_content(
            prompt
        )

        return response.text

    # =====================================================
    # BENCHMARK ANSWER COMPARISON
    # =====================================================
    def compare_answers(
        self,
        generated_answer,
        ground_truth
    ):

        prompt = f"""
You are an evaluator.

Compare the generated answer
against the ground truth answer.

====================================================

GENERATED ANSWER:
{generated_answer}

====================================================

GROUND TRUTH:
{ground_truth}

====================================================

Evaluate:
1. factual correctness
2. semantic similarity
3. completeness

Return ONLY a score between 0 and 1.

Example:
0.87
"""

        response = self.model.generate_content(
            prompt
        )

        try:
            return float(
                response.text.strip()
            )

        except:
            return 0.0