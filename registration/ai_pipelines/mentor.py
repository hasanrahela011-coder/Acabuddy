

import os

from google import genai
from google.genai import types


# ==========================================
# GEMINI CONFIGURATION
# ==========================================

MODEL_NAME = "gemini-2.5-flash"

api_key = os.getenv("GEMINI_API_KEY")

# Initialize the client only when the API key exists.
# This prevents Django from crashing during module import.
client = genai.Client(api_key=api_key) if api_key else None


# ==========================================
# MENTOR SYSTEM PROMPT
# ==========================================

SYSTEM_PROMPT = """
You are Acabuddy's Physics Mentor AI.

Your role is to guide students step by step
so they can understand and solve problems themselves.

RULES:
1. Do NOT immediately give the complete solution.
2. Break difficult problems into small, manageable steps.
3. Ask guiding questions that encourage critical thinking.
4. Explain concepts in simple, student-friendly language.
5. Encourage students to attempt the next step themselves.
6. Keep replies short, usually 3-6 lines.
7. Sound natural, supportive, and collaborative.
8. Adapt explanations to the student's current understanding.
9. If a student is stuck, give a small hint rather than the full answer.
10. Focus on helping the student learn, not merely obtain an answer.
"""


# ==========================================
# GEMINI RESPONSE FUNCTION
# ==========================================

def call_free_model(conversation):
    """
    Generate a short, step-by-step mentoring response
    using Google Gemini.
    """

    # Handle a missing API key without crashing the website.
    if client is None:
        return (
            "Mentor AI is temporarily unavailable. "
            "Please try again later."
        )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=conversation,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.6,
                max_output_tokens=400,
            ),
        )

        # Handle an empty model response.
        if not response.text:
            return (
                "I couldn't generate a response just now. "
                "Could you try asking your question again?"
            )

        return response.text

    except Exception as e:
        # Log the error for debugging without exposing API details
        # to the student.
        print(f"Acabuddy Mentor AI error: {type(e).__name__}: {e}")

        return (
            "I'm having trouble responding right now. "
            "Please try again in a moment."
        )
