import vertexai

from vertexai.generative_models import GenerativeModel


PROJECT_ID = "gd-gcp-gridu-genai"

vertexai.init(
    project=PROJECT_ID,
    location="us-central1"
)

model = GenerativeModel("gemini-2.0-flash")

response = model.generate_content(
    "Explain what IFC does in one sentence."
)

print(response.text)