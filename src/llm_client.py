"""
Thin wrapper around the Gemini API (Google Generative AI).

Everything that needs structured output goes through `call_tool`, which
forces the model to respond via a single function call (Gemini's function
calling, forced with tool_config mode="ANY") instead of free-text JSON.
"""
from __future__ import annotations

import json
import os
from typing import Any, Iterator

from src.config import GEMINI_API_KEY, GEMINI_MODEL, DEFAULT_TEMPERATURE

USE_MOCK_LLM = os.environ.get("USE_MOCK_LLM", "0") == "1"


class LLMConfigError(RuntimeError):
    """Raised when GEMINI_API_KEY is missing at call time."""


def _genai():
    if not GEMINI_API_KEY:
        raise LLMConfigError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and set your key, "
            "or `export GEMINI_API_KEY=...` before running (or in Colab: "
            "os.environ['GEMINI_API_KEY'] = '...')."
        )
    try:
        import google.generativeai as genai
    except ImportError as e:  # pragma: no cover
        raise LLMConfigError(
            "The 'google-generativeai' package is not installed. Run: pip install google-generativeai"
        ) from e
    genai.configure(api_key=GEMINI_API_KEY)
    return genai


def _to_gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively upper-cases JSON-schema `type` values (Gemini's protobuf
    Schema wants "OBJECT"/"STRING"/etc, not lowercase) and drops keys Gemini
    doesn't understand. Anthropic-style schemas otherwise map 1:1."""
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "type" and isinstance(value, str):
            out[key] = value.upper()
        elif key == "properties" and isinstance(value, dict):
            out[key] = {k: _to_gemini_schema(v) for k, v in value.items()}
        elif key == "items":
            out[key] = _to_gemini_schema(value)
        elif key in ("required", "enum", "description"):
            out[key] = value
        # silently drop anything else (e.g. "additionalProperties")
    return out


def call_tool(
    system_prompt: str,
    user_content: str,
    tool_schema: dict[str, Any],
    max_tokens: int = 1200,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict[str, Any]:
    """Calls the model, forcing it to respond via the given single function.
    Returns the parsed function-call args dict, in the same shape the rest
    of this repo expects (matches the old Anthropic tool_use.input shape).

    If USE_MOCK_LLM=1 is set, this bypasses the network entirely and returns
    a deterministic rule-based response (see src/mock_llm.py).
    """
    if USE_MOCK_LLM:
        from src.mock_llm import mock_call_tool
        return mock_call_tool(tool_schema["name"], user_content)

    genai = _genai()

    function_decl = {
        "name": tool_schema["name"],
        "description": tool_schema.get("description", ""),
        "parameters": _to_gemini_schema(tool_schema["input_schema"]),
    }

    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system_prompt,
        tools=[{"function_declarations": [function_decl]}],
        tool_config={
            "function_calling_config": {
                "mode": "ANY",
                "allowed_function_names": [tool_schema["name"]],
            }
        },
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        },
    )

    response = model.generate_content(user_content)

    for part in response.candidates[0].content.parts:
        fn = getattr(part, "function_call", None)
        if fn and fn.name == tool_schema["name"]:
            # fn.args is a protobuf Struct-like map; json round-trip gives
            # us a plain dict of plain Python types.
            return _to_native(dict(fn.args))

    raise ValueError(f"Model did not return the expected '{tool_schema['name']}' function call.")


def stream_text(
    system_prompt: str,
    user_content: str,
    max_tokens: int = 800,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Iterator[str]:
    """Yields text deltas as they arrive. Used for the streaming bonus
    (e.g. streaming the draft first-response message to the UI as it's
    generated, after the structured triage call has already completed)."""
    genai = _genai()
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system_prompt,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        },
    )
    response = model.generate_content(user_content, stream=True)
    for chunk in response:
        if chunk.text:
            yield chunk.text
def _to_native(obj):
    if hasattr(obj, "items"):
        return {k: _to_native(v) for k, v in obj.items()}
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
        return [_to_native(v) for v in obj]
    return obj
    
