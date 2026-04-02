"""Shared JSON extraction utilities for pipeline stages."""

from __future__ import annotations
import json
import re


def _strip_noise(text: str) -> str:
    """Strip <think> blocks and markdown code fences."""
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _balanced_extract(text: str, open_char: str, close_char: str) -> str | None:
    """Find the first balanced open_char...close_char substring, skipping strings."""
    start = text.find(open_char)
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i, c in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if c == "\\" and in_str:
            escape = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == open_char:
            depth += 1
        elif c == close_char:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json_obj(text: str) -> dict | None:
    """Extract the first balanced JSON object from LLM output."""
    text = _strip_noise(text)
    # Fast path: entire text is valid JSON
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    # Balanced extraction
    chunk = _balanced_extract(text, "{", "}")
    if chunk is None:
        return None
    try:
        return json.loads(chunk)
    except json.JSONDecodeError:
        return None


def extract_json_arr(text: str) -> list | None:
    """Extract the first balanced JSON array from LLM output."""
    text = _strip_noise(text)
    # Fast path
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    # Balanced extraction
    chunk = _balanced_extract(text, "[", "]")
    if chunk is None:
        return None
    try:
        return json.loads(chunk)
    except json.JSONDecodeError:
        return None
