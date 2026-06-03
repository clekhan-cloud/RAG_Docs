import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

print("API KEY:", api_key)

genai.configure(api_key=api_key)

model = genai.GenerativeModel(
    "gemini-2.0-flash"
)

response = model.generate_content(
    "What does IFC do?"
)

print(response.text)