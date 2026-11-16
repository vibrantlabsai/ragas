"""Shared utilities for project module."""

from __future__ import annotations

import random
from pathlib import Path


def validate_local_name(name: str) -> str:
    """Validate a dataset/experiment name used to build a local file path.

    A name is a single storage *identifier*, not a path. File-backed backends
    join it into a path (e.g. ``root/datasets/<name>.csv``), so a value with
    path semantics such as ``../../escape`` would resolve outside the intended
    storage directory. Reject names that are empty, refer to the current or
    parent directory, or contain path separators; ordinary identifiers such as
    ``dataset-1_v1.0`` pass through unchanged.

    Args:
        name: The dataset or experiment name to validate.

    Returns:
        The validated name, unchanged.

    Raises:
        ValueError: If the name is empty or contains path components.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Dataset/experiment name must be a non-empty string.")
    if name in (".", ".."):
        raise ValueError(
            f"Invalid dataset/experiment name {name!r}: refers to a directory."
        )
    if "/" in name or "\\" in name:
        raise ValueError(
            f"Invalid dataset/experiment name {name!r}: must not contain path "
            "separators."
        )
    # Defense in depth: the name must be a single, self-contained path component.
    if Path(name).name != name:
        raise ValueError(
            f"Invalid dataset/experiment name {name!r}: must be a simple "
            "identifier without path components."
        )
    return name


class MemorableNames:
    """Generator for memorable, unique names for experiments and datasets."""

    def __init__(self):
        # List of adjectives (similar to what Docker uses)
        self.adjectives = [
            "admiring",
            "adoring",
            "affectionate",
            "agitated",
            "amazing",
            "angry",
            "awesome",
            "blissful",
            "bold",
            "boring",
            "brave",
            "busy",
            "charming",
            "clever",
            "cool",
            "compassionate",
            "competent",
            "condescending",
            "confident",
            "cranky",
            "crazy",
            "dazzling",
            "determined",
            "distracted",
            "dreamy",
            "eager",
            "ecstatic",
            "elastic",
            "elated",
            "elegant",
            "eloquent",
            "epic",
            "fervent",
            "festive",
            "flamboyant",
            "focused",
            "friendly",
            "frosty",
            "gallant",
            "gifted",
            "goofy",
            "gracious",
            "happy",
            "hardcore",
            "heuristic",
            "hopeful",
            "hungry",
            "infallible",
            "inspiring",
            "jolly",
            "jovial",
            "keen",
            "kind",
            "laughing",
            "loving",
            "lucid",
            "magical",
            "mystifying",
            "modest",
            "musing",
            "naughty",
            "nervous",
            "nifty",
            "nostalgic",
            "objective",
            "optimistic",
            "peaceful",
            "pedantic",
            "pensive",
            "practical",
            "priceless",
            "quirky",
            "quizzical",
            "relaxed",
            "reverent",
            "romantic",
            "sad",
            "serene",
            "sharp",
            "silly",
            "sleepy",
            "stoic",
            "stupefied",
            "suspicious",
            "sweet",
            "tender",
            "thirsty",
            "trusting",
            "upbeat",
            "vibrant",
            "vigilant",
            "vigorous",
            "wizardly",
            "wonderful",
            "xenodochial",
            "youthful",
            "zealous",
            "zen",
        ]

        # List of influential computer scientists and tech entrepreneurs
        self.scientists = [
            "turing",
            "hopper",
            "knuth",
            "torvalds",
            "ritchie",
            "thompson",
            "dijkstra",
            "kay",
            "wozniak",
            "gates",
            "jobs",
            "musk",
            "bezos",
            "lovelace",
            "berners_lee",
            "cerf",
            "gosling",
            "kernighan",
            "lamport",
            "mccarthy",
            "minsky",
            "rossum",
            "backus",
            "engelbart",
            "hamilton",
            "chomsky",
            "shannon",
            "zuckerberg",
            "page",
            "brin",
            "matsumoto",
            "stallman",
            "stroustrup",
            "cook",
            "neumann",
            "babbage",
            "tanenbaum",
            "rivest",
            "shamir",
            "adleman",
            "carmack",
            "andreessen",
            "ullman",
            "postel",
            "huffman",
            "boole",
            "curry",
            "liskov",
            "wing",
            "goldwasser",
            "hoare",
            "milner",
            "perlis",
            "sutherland",
            "tarjan",
            "valiant",
            "yao",
            "hopcroft",
            "naur",
            "wilkes",
            "codd",
            "diffie",
            "hellman",
            "pearl",
            "thiel",
            "narayen",
            "nadella",
            "pichai",
            "dorsey",
        ]

        self.used_names = set()

    def generate_name(self):
        """Generate a single memorable name."""
        adjective = random.choice(self.adjectives)
        scientist = random.choice(self.scientists)
        return f"{adjective}_{scientist}"

    def generate_unique_name(self):
        """Generate a unique memorable name."""
        attempts = 0
        max_attempts = 100  # Prevent infinite loops

        while attempts < max_attempts:
            name = self.generate_name()
            if name not in self.used_names:
                self.used_names.add(name)
                return name
            attempts += 1

        # If we exhaust our combinations, add a random suffix
        base_name = self.generate_name()
        unique_name = f"{base_name}_{random.randint(1000, 9999)}"
        self.used_names.add(unique_name)
        return unique_name

    def generate_unique_names(self, count):
        """Generate multiple unique memorable names."""
        return [self.generate_unique_name() for _ in range(count)]


# Global instance for easy access
memorable_names = MemorableNames()
