import copy
import json
import re
import typing as t

from pydantic import BaseModel


def get_all_strings(obj: t.Any) -> list[str]:
    """
    Get all strings in the objects.
    """
    strings = []

    if isinstance(obj, str):
        strings.append(obj)
    elif isinstance(obj, BaseModel):
        for field_value in obj.model_dump().values():
            strings.extend(get_all_strings(field_value))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            strings.extend(get_all_strings(item))
    elif isinstance(obj, dict):
        for value in obj.values():
            strings.extend(get_all_strings(value))

    return strings


def update_strings(obj: t.Any, old_strings: list[str], new_strings: list[str]) -> t.Any:
    """
    Replace strings in the object with new strings.
    Example Usage:
    ```
    old_strings = ["old1", "old2", "old3"]
    new_strings = ["new1", "new2", "new3"]
    obj = {"a": "old1", "b": "old2", "c": ["old1", "old2", "old3"], "d": {"e": "old2"}}
    update_strings(obj, old_strings, new_strings)
    ```
    """
    if len(old_strings) != len(new_strings):
        raise ValueError("The number of old and new strings must be the same")

    def replace_string(s: str) -> str:
        for old, new in zip(old_strings, new_strings):
            if s == old:
                return new
        return s

    if isinstance(obj, str):
        return replace_string(obj)
    elif isinstance(obj, BaseModel):
        new_obj = copy.deepcopy(obj)
        for field in new_obj.__class__.model_fields:
            setattr(
                new_obj,
                field,
                update_strings(getattr(new_obj, field), old_strings, new_strings),
            )
        return new_obj
    elif isinstance(obj, list):
        return [update_strings(item, old_strings, new_strings) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(update_strings(item, old_strings, new_strings) for item in obj)
    elif isinstance(obj, dict):
        return {k: update_strings(v, old_strings, new_strings) for k, v in obj.items()}

    return copy.deepcopy(obj)


INJECTION_PATTERNS = [
    # System override directives: uppercase "SYSTEM OVERRIDE", or "[system override]", or "system override:" directive
    re.compile(r"\[\s*system\s+override\s*\]:?", flags=re.IGNORECASE),
    re.compile(r"\bSYSTEM\s+OVERRIDE\b:?"),
    re.compile(r"\bsystem\s+override\s*:", flags=re.IGNORECASE),
    re.compile(r"\bsystem\s+override\s*!", flags=re.IGNORECASE),
    # Instruction bypass directives: e.g. "ignore previous instructions", "disregard all prior instructions", "forget above prompts"
    re.compile(
        r"\b(?:ignore|disregard|forget)\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above)\s+(?:instructions|prompts|directives|commands)\b",
        flags=re.IGNORECASE,
    ),
    # New instruction directives: e.g. "new instructions:", "new instruction:"
    re.compile(r"\b(?:new\s+instructions?)\s*:", flags=re.IGNORECASE),
    # Role / control token boundaries: e.g. <system>, </system>, [INST], [/INST], <<SYS>>, <</SYS>>, <|im_start|>, <|im_end|>
    re.compile(r"<\/?(?:system|assistant|user|human)>", flags=re.IGNORECASE),
    re.compile(r"\[\/?(?:inst|sys)\]", flags=re.IGNORECASE),
    re.compile(r"<\/?(?:\|im_start\||\|im_end\|)>?", flags=re.IGNORECASE),
    re.compile(r"<<\/?SYS>>", flags=re.IGNORECASE),
]


def sanitize_context(text: str) -> str:
    """Sanitize retrieved context before embedding into judge prompts.

    Neutralizes prompt injection directives and control delimiters (e.g.
    'SYSTEM OVERRIDE', 'ignore previous instructions') by replacing them with
    '[SANITIZED]' while preserving legitimate context content.
    """
    if not isinstance(text, str) or not text:
        return text

    sanitized = text
    for pattern in INJECTION_PATTERNS:
        sanitized = pattern.sub("[SANITIZED]", sanitized)
    return sanitized


def extract_json(text: str) -> str:
    """Identify json from a text blob by matching '[]' or '{}'.

    Identifies and returns the LAST valid JSON structure from the text blob."""
    candidates: list[str] = []
    i = 0
    n = len(text)

    while i < n:
        if text[i] in ("{", "["):
            open_char = text[i]
            close_char = "}" if open_char == "{" else "]"
            count = 0
            start_idx = i
            in_string = False
            escape = False
            j = i
            while j < n:
                c = text[j]
                if escape:
                    escape = False
                elif c == "\\":
                    if in_string:
                        escape = True
                elif c == '"':
                    in_string = not in_string
                elif not in_string:
                    if c == open_char:
                        count += 1
                    elif c == close_char:
                        count -= 1
                        if count == 0:
                            candidates.append(text[start_idx : j + 1])
                            i = j
                            break
                j += 1
        i += 1

    # Filter for valid JSON structures using json.loads
    valid_candidates: list[str] = []
    for cand in candidates:
        try:
            json.loads(cand)
            valid_candidates.append(cand)
        except (ValueError, TypeError):
            continue

    if valid_candidates:
        return valid_candidates[-1]

    # Fallback to the last balanced candidate if none parsed via json.loads
    if candidates:
        return candidates[-1]

    return text  # In case of unbalanced JSON, return the original text
