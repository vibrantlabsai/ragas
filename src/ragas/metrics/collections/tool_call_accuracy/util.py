"""Tool Call Accuracy utility functions and models."""

import typing as t

from ragas.messages import ToolCall


def canonical_arg_value(value: t.Any) -> str:
    """Render an argument value into an order-independent string.

    ``str()`` on a dict reflects insertion order, so ``{"a": 1, "b": 2}`` and
    ``{"b": 2, "a": 1}`` would stringify differently even though they carry the
    same arguments. Sort mapping keys recursively so that nested structures
    canonicalize the same way top-level arguments already do.

    Sequences keep their order: element order is meaningful in a list.
    """
    if isinstance(value, t.Mapping):
        inner = ", ".join(
            f"{k!r}: {canonical_arg_value(value[k])}" for k in sorted(value, key=repr)
        )
        return "{" + inner + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(canonical_arg_value(item) for item in value) + "]"
    if isinstance(value, (set, frozenset)):
        return (
            "{" + ", ".join(sorted(canonical_arg_value(item) for item in value)) + "}"
        )
    return str(value)


def sorted_key_for_tool_call(tc: ToolCall) -> t.Tuple[str, ...]:
    """
    Generate a consistent sorting key for tool calls.

    Ensures tool calls with the same content are compared correctly
    regardless of argument order in the original call, including the key order
    of nested mappings.
    """
    key_list = [tc.name]
    args = tc.args
    args_names = sorted(args)
    for name in args_names:
        key_list.append(name)
        key_list.append(canonical_arg_value(args[name]))
    return tuple(key_list)


def exact_match_args(
    pred_args: t.Dict[str, t.Any], ref_args: t.Dict[str, t.Any]
) -> float:
    """Calculate exact match score for tool call arguments.

    Values are canonicalized before comparison so that nested mappings compare
    equal regardless of their key order.
    """
    if not ref_args and not pred_args:
        return 1.0
    if not ref_args:
        return 0.0

    score = 0.0
    for arg in ref_args.keys():
        if arg in pred_args and canonical_arg_value(
            pred_args[arg]
        ) == canonical_arg_value(ref_args[arg]):
            score += 1.0

    return score / len(ref_args)
