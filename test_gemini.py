from google import genai
from google.genai.types import HttpOptions

client = genai.Client(
    vertexai=True,
    project="gd-gcp-gridu-genai",
    location="us-central1",
    http_options=HttpOptions(api_version="v1")
)

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Hello Gemini"
)

print(response.text)