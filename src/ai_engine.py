"""
ai_engine.py
------------
The ONLY file that talks to Gemini. Everything upstream of this module
(data_engine, validators, analytics, rules) is deterministic and has already
computed real numbers. This file's entire job is to hand Gemini that
evidence, on a very short leash, and structurally validate what comes back.

Hard guarantees this module tries to enforce:
- Gemini is never given raw CSV rows or the ability to fetch more data - only
  the pre-computed `evidence` dict from analytics.build_evidence().
- The system prompt explicitly forbids inventing numbers/products/causes and
  requires the model to say when data is insufficient.
- The response is required to be JSON matching a fixed schema. If Gemini
  returns anything that doesn't parse/validate, OR the API key is missing,
  OR the API call fails for any reason, we return a safe HUMAN_REVIEW
  fallback object with the same schema shape - callers never have to handle
  a different response shape for the "AI failed" case.
"""

from __future__ import annotations
from typing import Dict, List, Optional
import json
import os
import re

from . import rules

GEMINI_MODEL_ENV = "GEMINI_MODEL"
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"

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


SYSTEM_INSTRUCTION = """You are StockSense AI, a retail sales & inventory reasoning assistant.

You are NOT a general chatbot. You reason ONLY over the structured JSON
"evidence" object you are given in each request. That evidence was produced
by deterministic pandas/numpy calculations BEFORE this conversation - it is
the complete and only source of truth available to you.

STRICT RULES (violating any of these makes your answer unusable):
1. Use ONLY numbers, product names, store names, and figures that literally
   appear in the supplied evidence JSON. Never invent, estimate, round in a
   misleading way, or "fill in" a number that is missing or null.
2. If a field in the evidence is null, missing, or explicitly marked as
   unknown/insufficient (e.g. "UNKNOWN_NO_SALES_VELOCITY", None, or absent),
   you MUST say the data is insufficient for that specific point rather than
   guessing. Do not silently skip it either - name what's missing.
2a. Any reorder level marked "reorder_level_is_derived_estimate": true is an
    ASSUMPTION (a rule-based estimate), not a fact from the retailer's data.
    Label it as such if you reference it.
3. Never invent a root cause for a trend, spike, drop, or stock-out. You may
   only state that a pattern EXISTS in the data (e.g. "sales dropped 60% vs
   baseline"); you must NOT claim WHY it happened unless the evidence
   contains an explicit causal field.
4. Never invent or assume a product, store, or category that is not present
   in the evidence. If the user asked about something not found in the
   evidence, say so plainly.
5. Clearly separate FACTS (directly from evidence) from ASSUMPTIONS (any
   inference, default, or derived estimate, including default reorder
   levels). Facts go implicitly into "answer"/"key_finding"/"evidence" (as
   literal figures); anything you reasoned beyond the literal numbers goes
   into "assumptions".
6. If the evidence is too sparse, ambiguous, or ambiguous-entity (e.g. the
   product/store the user asked about was not found) to answer responsibly,
   set "escalation" to "HUMAN_REVIEW" and set "confidence" to "LOW".
7. You MUST reply with ONLY a single JSON object, no markdown fences, no
   prose outside the JSON, matching EXACTLY this schema:
{
  "answer": "<direct, plain-language answer to the user's question>",
  "key_finding": "<the single most important fact/insight from the evidence>",
  "recommendation": "<a concrete, actionable next step for a human decision-maker>",
  "evidence": ["<short literal facts/figures pulled from the evidence, one per item>"],
  "assumptions": ["<any estimate, default, or inference you made, one per item; empty array if none>"],
  "confidence": "HIGH|MEDIUM|LOW",
  "escalation": "NONE|HUMAN_REVIEW"
}
8. Set "confidence" honestly based on data completeness/recency, not on how
   confident your prose sounds. Sparse history, missing fields, or an
   "UNKNOWN"-style risk label should push confidence toward LOW/MEDIUM.
9. "recommendation" must be a human-actionable suggestion (e.g. "reorder X
   units", "review store Y's stock", "escalate to category manager") - never
   a recommendation to "consult more data" as a way of avoiding the
   question when the evidence already supports an answer.
"""


class GeminiUnavailableError(Exception):
    pass


def _fallback_response(reason: str, confidence: str = "LOW") -> Dict:
    return {
        "answer": f"Unable to produce a Gemini-grounded answer: {reason}",
        "key_finding": "AI reasoning layer could not complete this request.",
        "recommendation": "A human should review the underlying evidence directly before acting.",
        "evidence": [],
        "assumptions": [],
        "confidence": confidence,
        "escalation": "HUMAN_REVIEW",
    }


def _get_api_key() -> Optional[str]:
    return os.environ.get("GEMINI_API_KEY")


def _extract_json(text: str) -> Optional[Dict]:
    """Best-effort extraction of a JSON object from a model response, tolerating
    stray markdown fences or leading/trailing prose the model might add."""
    if not text:
        return None
    text = text.strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text.strip()).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to grabbing the first {...} block
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _validate_schema(obj: Dict) -> List[str]:
    """Returns a list of validation problems; empty list means valid."""
    problems = []
    if not isinstance(obj, dict):
        return ["Response is not a JSON object."]
    for key, expected_type in RESPONSE_SCHEMA_KEYS.items():
        if key not in obj:
            problems.append(f"Missing key '{key}'.")
            continue
        if not isinstance(obj[key], expected_type):
            problems.append(f"Key '{key}' should be {expected_type.__name__}.")
    if "confidence" in obj and obj.get("confidence") not in VALID_CONFIDENCE:
        problems.append("'confidence' must be one of HIGH|MEDIUM|LOW.")
    if "escalation" in obj and obj.get("escalation") not in VALID_ESCALATION:
        problems.append("'escalation' must be one of NONE|HUMAN_REVIEW.")
    return problems


def _build_user_prompt(question: str, evidence: Dict, data_quality_notes: Optional[List[str]] = None) -> str:
    payload = {
        "user_question": question,
        "evidence": evidence,
        "data_quality_notes": data_quality_notes or [],
    }
    return (
        "Here is the user's question and the ONLY evidence you may reason over. "
        "Respond with the required JSON schema only.\n\n"
        + json.dumps(payload, default=str, indent=2)
    )


def _call_gemini(system_instruction: str, user_prompt: str, model_name: str) -> str:
    """Isolated so it's the single point that touches the google-generativeai SDK."""
    import google.generativeai as genai

    api_key = _get_api_key()
    if not api_key:
        raise GeminiUnavailableError("GEMINI_API_KEY is not set in the environment.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name=model_name, system_instruction=system_instruction)

    generation_config = genai.types.GenerationConfig(
        temperature=0.1,
        response_mime_type="application/json",
    )

    response = model.generate_content(user_prompt, generation_config=generation_config)

    text = getattr(response, "text", None)
    if not text:
        # Some SDK versions require pulling text out of candidates/parts.
        try:
            text = response.candidates[0].content.parts[0].text
        except Exception:  # noqa: BLE001
            text = None
    if not text:
        raise GeminiUnavailableError("Gemini returned an empty response (possibly blocked by safety filters).")
    return text


def ask_gemini(question: str, evidence: Dict, data_quality_notes: Optional[List[str]] = None) -> Dict:
    """
    Main entry point used by app.py.

    Returns a dict that ALWAYS matches RESPONSE_SCHEMA_KEYS, regardless of
    whether Gemini succeeded, failed, or returned malformed output.
    """
    if not question or not question.strip():
        return _fallback_response("No question was provided.")

    if evidence is None:
        return _fallback_response("No evidence was available to reason over.")

    api_key = _get_api_key()
    if not api_key:
        return _fallback_response(
            "GEMINI_API_KEY is not configured on the server. Set it in your .env file to enable AI reasoning."
        )

    model_name = os.environ.get(GEMINI_MODEL_ENV, DEFAULT_GEMINI_MODEL)
    user_prompt = _build_user_prompt(question, evidence, data_quality_notes)

    try:
        raw_text = _call_gemini(SYSTEM_INSTRUCTION, user_prompt, model_name)
    except GeminiUnavailableError as e:
        return _fallback_response(f"Gemini is unavailable: {e}")
    except Exception as e:  # noqa: BLE001 - network errors, quota errors, SDK errors, etc.
        return _fallback_response(f"Gemini call failed unexpectedly: {e}")

    parsed = _extract_json(raw_text)
    if parsed is None:
        return _fallback_response("Gemini's response could not be parsed as JSON.")

    problems = _validate_schema(parsed)
    if problems:
        return _fallback_response(
            "Gemini's response did not match the required schema: " + "; ".join(problems)
        )

    # Normalize: ensure list items are strings (defensive against odd model output)
    parsed["evidence"] = [str(x) for x in parsed.get("evidence", [])]
    parsed["assumptions"] = [str(x) for x in parsed.get("assumptions", [])]

    return parsed
