"""
ai_engine.py

The ONLY file that talks to Gemini.

Everything upstream of this module (data_engine, validators, analytics,
rules) is deterministic and has already computed real numbers.

This file:
1. Sends only pre-computed evidence to Gemini.
2. Prevents unsupported claims through strict instructions.
3. Requires a fixed JSON response schema.
4. Validates the returned response.
5. Returns a safe HUMAN_REVIEW fallback if Gemini fails.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import json
import os
import re

from . import rules


GEMINI_MODEL_ENV = "GEMINI_MODEL"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


RESPONSE_SCHEMA_KEYS = {
    "answer": str,
    "key_finding": str,
    "recommendation": str,
    "evidence": list,
    "assumptions": list,
    "confidence": str,
    "escalation": str,
}

VALID_CONFIDENCE = {"HIGH", "MEDIUM", "LOW"}
VALID_ESCALATION = {"NONE", "HUMAN_REVIEW"}


SYSTEM_INSTRUCTION = """
You are StockSense AI, a retail sales and inventory reasoning assistant.

You are NOT a general chatbot.

You must reason ONLY over the structured JSON "evidence" object provided
in each request.

The evidence was produced by deterministic pandas/numpy calculations
before this conversation. It is the complete and only source of truth.

STRICT RULES:

1. Use ONLY numbers, product names, store names, categories, and figures
   that literally appear in the supplied evidence JSON.

2. Never invent, estimate, or guess a missing number.

3. If a field is null, missing, unknown, or explicitly marked as insufficient,
   say that the data is insufficient for that specific point.

4. Any reorder level marked as:
   "reorder_level_is_derived_estimate": true
   is an ASSUMPTION, not a fact from the retailer's data.
   Clearly label it as an assumption if referenced.

5. Never invent a root cause for a trend, spike, drop, or stock-out.

   You may describe a pattern that exists in the evidence.

   You must NOT claim WHY something happened unless an explicit causal
   field exists in the evidence.

6. Never invent or assume a product, store, supplier, or category that is
   not present in the evidence.

7. Clearly separate facts from assumptions.

8. If the evidence is too sparse, ambiguous, or does not contain the entity
   requested by the user:

   - confidence = "LOW"
   - escalation = "HUMAN_REVIEW"

9. The response MUST be exactly one JSON object.

10. Do NOT return markdown.
    Do NOT return ```json.
    Do NOT return explanatory text outside the JSON.

11. The JSON MUST match exactly this schema:

{
  "answer": "direct plain-language answer",
  "key_finding": "single most important fact",
  "recommendation": "concrete actionable human next step",
  "evidence": [
    "short literal facts from evidence"
  ],
  "assumptions": [
    "estimates or inferences; empty if none"
  ],
  "confidence": "HIGH|MEDIUM|LOW",
  "escalation": "NONE|HUMAN_REVIEW"
}

12. Confidence must reflect actual data completeness and recency.

13. Recommendations must be actionable for a human decision-maker.

Examples:

- "Review the critical-risk products for replenishment."
- "Reorder the affected product based on the available inventory evidence."
- "Escalate the item to the category manager for review."

Never recommend "get more data" when the available evidence already
supports a reasonable action.

If the question asks for a cause that cannot be established from the
evidence, explicitly say that the cause cannot be determined from the
available evidence and escalate to HUMAN_REVIEW.
"""


class GeminiUnavailableError(Exception):
    """Raised when Gemini cannot be reached or configured correctly."""


def _fallback_response(reason: str, confidence: str = "LOW") -> Dict:
    """
    Always return the same safe response structure when Gemini fails.
    """

    return {
        "answer": f"Unable to produce a Gemini-grounded answer: {reason}",
        "key_finding": "AI reasoning layer could not complete this request.",
        "recommendation": (
            "A human should review the underlying evidence directly "
            "before acting."
        ),
        "evidence": [],
        "assumptions": [],
        "confidence": confidence,
        "escalation": "HUMAN_REVIEW",
    }


def _get_api_key() -> Optional[str]:
    """Read Gemini API key from environment."""

    key = os.environ.get("GEMINI_API_KEY")

    if key:
        return key.strip()

    return None


def _extract_json(text: str) -> Optional[Dict]:
    """
    Best-effort JSON extraction.

    Handles:
    - pure JSON
    - ```json ... ```
    - accidental surrounding text
    """

    if not text:
        return None

    text = text.strip()

    # Remove markdown code fences.
    text = re.sub(
        r"^```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    text = re.sub(
        r"```$",
        "",
        text,
    ).strip()

    # First try the entire response.
    try:
        parsed = json.loads(text)

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        pass

    # Try extracting the first JSON object.
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:

        candidate = text[start : end + 1]

        try:
            parsed = json.loads(candidate)

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            pass

    return None


def _validate_schema(obj: Dict) -> List[str]:
    """
    Validate the fixed response schema.

    Returns a list of problems.
    Empty list means valid.
    """

    problems: List[str] = []

    if not isinstance(obj, dict):
        return ["Response is not a JSON object."]

    for key, expected_type in RESPONSE_SCHEMA_KEYS.items():

        if key not in obj:
            problems.append(
                f"Missing key '{key}'."
            )
            continue

        if not isinstance(obj[key], expected_type):
            problems.append(
                f"Key '{key}' should be {expected_type.__name__}."
            )

    if (
        "confidence" in obj
        and obj.get("confidence") not in VALID_CONFIDENCE
    ):
        problems.append(
            "'confidence' must be one of HIGH|MEDIUM|LOW."
        )

    if (
        "escalation" in obj
        and obj.get("escalation") not in VALID_ESCALATION
    ):
        problems.append(
            "'escalation' must be one of NONE|HUMAN_REVIEW."
        )

    return problems


def _build_user_prompt(
    question: str,
    evidence: Dict,
    data_quality_notes: Optional[List[str]] = None,
) -> str:
    """
    Build the only user prompt sent to Gemini.

    Raw CSV rows are never sent.
    """

    payload = {
        "user_question": question,
        "evidence": evidence,
        "data_quality_notes": data_quality_notes or [],
    }

    return (
        "Here is the user's question and the ONLY evidence you may "
        "reason over.\n"
        "Respond with the required JSON schema only.\n\n"
        + json.dumps(
            payload,
            default=str,
            indent=2,
        )
    )


def _call_gemini(
    system_instruction: str,
    user_prompt: str,
    model_name: str,
) -> str:
    """
    Single point that communicates with the Google Gemini SDK.

    Uses the modern google-genai SDK.
    """

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise GeminiUnavailableError(
            "google-genai package is not installed. "
            "Run: pip install -r requirements.txt"
        ) from exc

    api_key = _get_api_key()

    if not api_key:
        raise GeminiUnavailableError(
            "GEMINI_API_KEY is not set in the environment."
        )

    try:
        client = genai.Client(
            api_key=api_key
        )

        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
            ),
        )

    except Exception as exc:
        raise GeminiUnavailableError(
            f"Gemini API request failed: {exc}"
        ) from exc

    text = getattr(response, "text", None)

    if text:
        return text.strip()

    # Defensive fallback for SDK response structures.
    try:
        candidates = getattr(response, "candidates", [])

        if candidates:

            content = getattr(
                candidates[0],
                "content",
                None,
            )

            parts = getattr(
                content,
                "parts",
                [],
            )

            if parts:

                part_text = getattr(
                    parts[0],
                    "text",
                    None,
                )

                if part_text:
                    return part_text.strip()

    except Exception:
        pass

    raise GeminiUnavailableError(
        "Gemini returned an empty response."
    )


def ask_gemini(
    question: str,
    evidence: Dict,
    data_quality_notes: Optional[List[str]] = None,
) -> Dict:
    """
    Main entry point used by app.py.

    Always returns the same response schema, even if Gemini fails.
    """

    # Validate question.
    if not question or not question.strip():

        return _fallback_response(
            "No question was provided."
        )

    # Validate evidence.
    if evidence is None:

        return _fallback_response(
            "No evidence was available to reason over."
        )

    # Validate API key.
    api_key = _get_api_key()

    if not api_key:

        return _fallback_response(
            "GEMINI_API_KEY is not configured on the server. "
            "Set it in your .env file to enable AI reasoning."
        )

    # Select model.
    model_name = os.environ.get(
        GEMINI_MODEL_ENV,
        DEFAULT_GEMINI_MODEL,
    ).strip()

    if not model_name:
        model_name = DEFAULT_GEMINI_MODEL

    # Build evidence-only prompt.
    user_prompt = _build_user_prompt(
        question=question,
        evidence=evidence,
        data_quality_notes=data_quality_notes,
    )

    # Call Gemini.
    try:

        raw_text = _call_gemini(
            SYSTEM_INSTRUCTION,
            user_prompt,
            model_name,
        )

    except GeminiUnavailableError as exc:

        return _fallback_response(
            f"Gemini is unavailable: {exc}"
        )

    except Exception as exc:

        return _fallback_response(
            f"Gemini call failed unexpectedly: {exc}"
        )

    # Parse JSON.
    parsed = _extract_json(raw_text)

    if parsed is None:

        return _fallback_response(
            "Gemini's response could not be parsed as JSON."
        )

    # Validate schema.
    problems = _validate_schema(parsed)

    if problems:

        return _fallback_response(
            "Gemini's response did not match the required schema: "
            + "; ".join(problems)
        )

    # Defensive normalization.
    parsed["evidence"] = [
        str(item)
        for item in parsed.get("evidence", [])
    ]

    parsed["assumptions"] = [
        str(item)
        for item in parsed.get("assumptions", [])
    ]

    return parsed