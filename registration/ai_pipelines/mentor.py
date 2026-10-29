
import os
from google import genai
from google.genai import types

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

MODEL_NAME = "gemini-2.5-flash"

SYSTEM_PROMPT = """
You are a physics mentor guiding a student step by step.

Rules:
- Do NOT give the full solution.
- Break problems into micro steps.
- Ask guiding questions.
- Encourage thinking.
- Keep replies short (3-6 lines).
- Sound natural and collaborative.
"""

def call_free_model(conversation):
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=conversation,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.6,
            max_output_tokens=400,
        ),
    )

    return response.text
