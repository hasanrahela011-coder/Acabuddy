import os

from google import genai
from google.genai import types


# ==========================================
# GEMINI CONFIGURATION
# ==========================================

MODEL_NAME = "gemini-3.8-flash"

api_key = os.getenv("GEMINI_API_KEY_MENTOR")

# Initialize the client only when the API key exists.
# This prevents Django from crashing during module import.
client = genai.Client(api_key=api_key) if api_key else None


# ==========================================
# MENTOR SYSTEM PROMPT
# ==========================================

SYSTEM_PROMPT = """
You are Acabuddy's Mentor AI.

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

    if client is None:
        return "Mentor AI is unavailable. Please check the API key."

    try:
        gemini_history = []

        for message in conversation:
            role = message.get("role")
            content = message.get("content", "")

            # System prompt is sent separately
            if role == "system":
                continue

            # Convert assistant role to Gemini model role
            if role == "assistant":
                role = "model"

            if role not in ("user", "model"):
                continue

            if not isinstance(content, str):
                content = str(content)

            if not content.strip():
                continue

            # Gemini-compatible message structure
            gemini_history.append({
                "role": role,
                "parts": [
                    {"text": content}
                ]
            })

        if not gemini_history:
            return "Please ask me a question so I can help."

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=gemini_history,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "temperature": 0.6,
                "max_output_tokens": 400,
            }
        )

        if response.text:
            return response.text

        return "I couldn't generate a response. Please try again."

    except Exception as e:
        print(
            f"Acabuddy Mentor AI error: "
            f"{type(e).__name__}: {e}"
        )

        return (
            "I'm having trouble responding right now. "
            "Please try again in a moment."
        )
