import copy
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


def extract_json(text: str) -> str:
    """Identify json from a text blob by matching '[]' or '{}'.

    Extracts the LAST complete JSON object, not the first. The LLM-judge's own
    verdict is the authoritative JSON and appears last in the response; JSON that
    appeared earlier may have originated from the model-under-test's output (which
    is embedded in the judge prompt) and was referenced in the judge's reasoning.
    Extracting the first JSON object allowed verdict injection.
    """

    # check for markdown indicator; if present, start there
    md_json_idx = text.find("```json")
    if md_json_idx != -1:
        text = text[md_json_idx:]

    # Find ALL complete JSON objects, return the LAST one.
    # Walk forward tracking brace/bracket depth, collecting complete objects.
    objects: list[str] = []
    i = 0
    while i < len(text):
        if text[i] in "{[":
            open_char = text[i]
            close_char = "]" if open_char == "[" else "}"
            depth = 0
            in_string = False
            escape = False
            for j in range(i, len(text)):
                ch = text[j]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == open_char:
                    depth += 1
                elif ch == close_char:
                    depth -= 1
                    if depth == 0:
                        objects.append(text[i : j + 1])
                        i = j + 1
                        break
            else:
                # Unbalanced — incomplete JSON, skip
                break
        else:
            i += 1

    if objects:
        return objects[-1]  # the LAST complete JSON object

    # Fallback: original behavior for backward compat (unbalanced/incomplete JSON)
    left_bracket_idx = text.find("[")
    left_brace_idx = text.find("{")
    indices = [idx for idx in (left_bracket_idx, left_brace_idx) if idx != -1]
    start_idx = min(indices) if indices else None
    if start_idx is None:
        return text
    open_char = text[start_idx]
    close_char = "]" if open_char == "[" else "}"
    count = 0
    for i, char in enumerate(text[start_idx:], start=start_idx):
        if char == open_char:
            count += 1
        elif char == close_char:
            count -= 1
        if count == 0:
            return text[start_idx : i + 1]
    return text
