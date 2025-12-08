"""Tool Call Accuracy utility functions and models."""

import typing as t

from ragas.messages import ToolCall


def sorted_key_for_tool_call(tc: ToolCall) -> t.Tuple[str, ...]:
    """
    Generate a consistent sorting key for tool calls.

    Ensures tool calls with the same content are compared correctly
    regardless of argument order in the original call.
    """
    key_list = [tc.name]
    args = tc.args
    args_names = sorted(args)
    for name in args_names:
        key_list.append(name)
        key_list.append(str(args[name]))
    return tuple(key_list)


def exact_match_args(
    pred_args: t.Dict[str, t.Any], ref_args: t.Dict[str, t.Any]
) -> float:
    """Calculate exact match score for tool call arguments."""
    if not ref_args and not pred_args:
        return 1.0
    if not ref_args:
        return 0.0

    score = 0.0
    for arg in ref_args.keys():
        if arg in pred_args and str(pred_args[arg]) == str(ref_args[arg]):
            score += 1.0

    return score / len(ref_args)
