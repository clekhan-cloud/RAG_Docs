import vertexai
from vertexai.generative_models import GenerativeModel

vertexai.init(
    project="YOUR_PROJECT_ID",
    location="us-central1"
)

model = GenerativeModel("gemini-2.0-flash")

response = model.generate_content("Hello")

print(response.text)