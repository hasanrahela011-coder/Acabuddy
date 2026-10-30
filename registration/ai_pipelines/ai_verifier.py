
import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Gemini models
SMART_MODEL = "gemini-2.5-flash"
FAST_MODEL = "gemini-2.5-flash"

# Read the API key from the environment.
api_key = os.getenv("GEMINI_API_KEY_VERIFIER")

# Avoid crashing Django during module import if the key is missing.
client = genai.Client(api_key=api_key) if api_key else None


def call_llm(prompt, system_prompt, model, json_mode=False):
    """
    Wrapper for Gemini API calls.
    Supports system instructions and JSON output.
    """
    if client is None:
        return "Error: GEMINI_API_KEY environment variable is not set."

    try:
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2 if json_mode else 0.0,
            response_mime_type="application/json" if json_mode else "text/plain",
        )

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

        if response.text:
            return response.text

        return "Error: Gemini returned an empty response."

    except Exception as e:
        return f"Error: {str(e)}"


# --- AGENT 1: THE SOLVER ---
def solver(question, domain):
    system = f"""
    You are an expert professional in {domain}.

    Your task is to solve the user's problem accurately and concisely,
    following modern best practices.

    1. Rely on your internal knowledge.
    2. Do not use any provided candidate answer to generate the reference.
    3. Be concise and provide code or steps when applicable.
    4. For subjective questions, give the generally accepted academic view.
    """

    return call_llm(question, system, SMART_MODEL)


# --- AGENT 2: THE COMPARATOR ---
def comparator(question, reference, target):
    system = """
    You are a meticulous technical auditor.

    Compare the Reference Solution with the Candidate Answer.
    Determine whether the candidate is functionally equivalent.

    INSTRUCTIONS:
    1. If both convey the same meaning and logic, output:
       NO SEMANTIC DIFFERENCES.
    2. If there are logical errors or missing facts, list them as bullets.
    3. Ignore tone, formatting, and variable naming.
    4. If the candidate uses a different but valid approach,
       explicitly identify it as valid.
    """

    user_input = f"""
    [QUESTION]:
    {question}

    [REFERENCE SOLUTION]:
    {reference}

    [CANDIDATE ANSWER]:
    {target}
    """

    return call_llm(user_input, system, SMART_MODEL)


# --- AGENT 3: THE CRITIC ---
def critic(question, comparison, target):
    system = """
    You are an expert domain grader.

    Score the Candidate Answer based on factual accuracy and the Audit Report.

    INSTRUCTIONS:
    1. Prioritize meaning over wording.
    2. Ignore stylistic differences unless they change the meaning.
    3. Identify the core facts in the reference solution.
    4. Check whether those facts appear in the candidate answer.

    SCORING:
    - 1.0: Correct and semantically equivalent.
    - 0.8: Correct with minor omissions.
    - 0.5: Partially correct; core concepts are missing.
    - 0.0: Incorrect or dangerously wrong.

    Return ONLY a valid JSON object with this structure:
    {
      "score": 0.0,
      "reasoning": "Concise explanation of factual accuracy"
    }
    """

    user_input = f"""
    [QUESTION]:
    {question}

    [CANDIDATE ANSWER]:
    {target}

    [AUDIT REPORT]:
    {comparison}
    """

    return call_llm(
        user_input,
        system,
        FAST_MODEL,
        json_mode=True,
    )


# --- MAIN WORKFLOW ---
def verify_answer(question: str, answer_text: str, domain: str):
    # Step 1: Generate a reference solution.
    reference_sol = solver(question, domain)

    # Stop if the reference generation failed.
    if reference_sol.startswith("Error:"):
        return {
            "score": 0.0,
            "reasoning": reference_sol,
            "audit": "",
            "verified": False,
        }

    # Step 2: Compare the reference with the candidate.
    diffs = comparator(question, reference_sol, answer_text)

    if diffs.startswith("Error:"):
        return {
            "score": 0.0,
            "reasoning": diffs,
            "audit": "",
            "verified": False,
        }

    # Step 3: Grade the candidate.
    verdict_json = critic(question, diffs, answer_text)

    if verdict_json.startswith("Error:"):
        return {
            "score": 0.0,
            "reasoning": verdict_json,
            "audit": diffs,
            "verified": False,
        }

    try:
        # Handle JSON enclosed in Markdown code fences, if returned.
        cleaned = verdict_json.strip()

        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            if cleaned.rstrip().endswith("```"):
                cleaned = cleaned.rstrip()[:-3].strip()

        result = json.loads(cleaned)
        score = float(result["score"])

        # Validate the returned score.
        if not 0.0 <= score <= 1.0:
            raise ValueError("Score must be between 0.0 and 1.0.")

        return {
            "score": score,
            "reasoning": str(result.get("reasoning", "")),
            "audit": diffs,
            "verified": score >= 0.6,
        }

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        return {
            "score": 0.0,
            "reasoning": f"AI verification failed: {str(e)}",
            "audit": verdict_json,
            "verified": False,
        }

