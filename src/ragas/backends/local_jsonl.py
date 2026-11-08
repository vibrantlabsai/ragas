"""Local JSONL backend implementation for projects and datasets."""

import json
import re
import typing as t
from datetime import date, datetime
from pathlib import Path

from pydantic import BaseModel

from .base import BaseBackend

# Discriminator used to tag values that were Python date / datetime objects at
# save time. Only tagged values are restored to date / datetime on load, so
# user strings that merely look like ISO 8601 timestamps round-trip unchanged.
_RAGAS_TYPE_KEY = "__ragas_type__"

# Legacy fallback (files written before tagging existed): only coerce strings
# whose *entire* value matches an anchored ISO 8601 date or datetime shape.
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ISO_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$"
)


class LocalJSONLBackend(BaseBackend):
    """File-based backend using JSONL format for local storage.

    Stores datasets and experiments as JSONL files (one JSON object per line).
    Preserves data types and supports complex nested structures including
    datetime objects, lists, and nested dictionaries.

    Directory Structure:
        root_dir/
        ├── datasets/
        │   ├── dataset1.jsonl
        │   └── dataset2.jsonl
        └── experiments/
            ├── experiment1.jsonl
            └── experiment2.jsonl

    Args:
        root_dir: Directory path for storing JSONL files

    Features:
        - Preserves Python data types (int, float, bool, None)
        - Automatic datetime/date round trip via a ``__ragas_type__``
          discriminator (date-like strings are never coerced)
        - Supports nested dictionaries and lists
        - Handles malformed JSON lines gracefully (skips with warning)
        - UTF-8 encoding for international text
        - Compact JSON formatting (no extra whitespace)

    Best For:
        - Complex data structures with nesting
        - Mixed data types and datetime objects
        - When data type preservation is important
        - Large datasets (streaming line-by-line processing)
    """

    def __init__(
        self,
        root_dir: str,
    ):
        self.root_dir = Path(root_dir)

    def _get_data_dir(self, data_type: str) -> Path:
        """Get the directory path for datasets or experiments."""
        return self.root_dir / data_type

    def _get_file_path(self, data_type: str, name: str) -> Path:
        """Get the full file path for a dataset or experiment."""
        return self._get_data_dir(data_type) / f"{name}.jsonl"

    def _serialize_datetime(self, obj: t.Any, tagged: bool = True) -> t.Any:
        """Serialize date / datetime objects to tagged dicts.

        Typed values are written as ``{"__ragas_type__": "date" | "datetime",
        "value": "<iso>"}`` so the load path can restore exactly the values
        that were date / datetime objects at save time. Strings that would be
        mistaken for a legacy ISO value are written with a ``"str"`` tag, and
        user dicts that already contain the discriminator key are wrapped with
        a ``"dict"`` tag so they round trip verbatim.
        """
        if isinstance(obj, datetime):
            if not tagged:
                return obj.isoformat()
            return {_RAGAS_TYPE_KEY: "datetime", "value": obj.isoformat()}
        elif isinstance(obj, date):
            if not tagged:
                return obj.isoformat()
            return {_RAGAS_TYPE_KEY: "date", "value": obj.isoformat()}
        elif isinstance(obj, str) and tagged and self._is_legacy_iso_string(obj):
            return {_RAGAS_TYPE_KEY: "str", "value": obj}
        elif isinstance(obj, dict):
            if tagged and _RAGAS_TYPE_KEY in obj:
                # Never re-interpret a user dict that carries the
                # discriminator key: wrap it (inner values serialized tag-free)
                # so it round trips verbatim instead of being restored as a
                # date / datetime / string on load.
                return {
                    _RAGAS_TYPE_KEY: "dict",
                    "value": {
                        k: self._serialize_datetime(v, tagged=False)
                        for k, v in obj.items()
                    },
                }
            return {k: self._serialize_datetime(v, tagged) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_datetime(item, tagged) for item in obj]
        else:
            return obj

    @staticmethod
    def _is_legacy_iso_string(obj: str) -> bool:
        """Return True for strings whose whole value looks like a legacy ISO date/datetime."""
        return bool(_ISO_DATE_RE.match(obj) or _ISO_DATETIME_RE.match(obj))

    def _deserialize_datetime(self, obj: t.Any) -> t.Any:
        """Restore tagged values to date / datetime; keep everything else as-is.

        Only dicts carrying the ``__ragas_type__`` discriminator written by the
        save path are converted back to date / datetime. For legacy records
        (written before tagging existed), strings are coerced only when their
        whole value matches an anchored ISO 8601 date or datetime shape.
        """
        if isinstance(obj, str):
            if _ISO_DATETIME_RE.match(obj):
                try:
                    return datetime.fromisoformat(obj.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    return obj
            if _ISO_DATE_RE.match(obj):
                try:
                    return datetime.fromisoformat(obj + "T00:00:00").date()
                except (ValueError, TypeError):
                    return obj
            return obj
        elif isinstance(obj, dict):
            tag = obj.get(_RAGAS_TYPE_KEY)
            if set(obj) == {_RAGAS_TYPE_KEY, "value"}:
                value = obj["value"]
                if tag == "str":
                    return (
                        value
                        if isinstance(value, str)
                        else self._deserialize_datetime(value)
                    )
                if tag == "dict":
                    # Written verbatim by the save path; restore verbatim.
                    return value
                if tag in ("date", "datetime") and isinstance(value, str):
                    try:
                        if tag == "datetime":
                            return datetime.fromisoformat(value.replace("Z", "+00:00"))
                        return datetime.fromisoformat(value + "T00:00:00").date()
                    except (ValueError, TypeError):
                        return obj
            return {k: self._deserialize_datetime(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deserialize_datetime(item) for item in obj]
        else:
            return obj

    def _load(self, data_type: str, name: str) -> t.List[t.Dict[str, t.Any]]:
        """Load data from JSONL file, raising FileNotFoundError if file doesn't exist."""
        file_path = self._get_file_path(data_type, name)

        if not file_path.exists():
            raise FileNotFoundError(
                f"No {data_type[:-1]} named '{name}' found at {file_path}"
            )

        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                try:
                    # Parse JSON line
                    json_obj = json.loads(line)
                    # Deserialize datetime objects
                    json_obj = self._deserialize_datetime(json_obj)
                    data.append(json_obj)
                except json.JSONDecodeError as e:
                    # Handle malformed JSON gracefully
                    print(f"Warning: Skipping malformed JSON on line {line_num}: {e}")
                    continue

        return data

    def _save(
        self,
        data_type: str,
        name: str,
        data: t.List[t.Dict[str, t.Any]],
        data_model: t.Optional[t.Type[BaseModel]],
    ) -> None:
        """Save data to JSONL file, creating directory if needed."""
        file_path = self._get_file_path(data_type, name)

        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Handle empty data
        if not data:
            # Create empty JSONL file
            with open(file_path, "w", encoding="utf-8") as f:
                pass
            return

        # Write data to JSONL
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                # Serialize datetime objects
                serialized_item = self._serialize_datetime(item)
                # Write as JSON line
                json_line = json.dumps(
                    serialized_item, ensure_ascii=False, separators=(",", ":")
                )
                f.write(json_line + "\n")

    def _list(self, data_type: str) -> t.List[str]:
        """List all available datasets or experiments."""
        data_dir = self._get_data_dir(data_type)

        if not data_dir.exists():
            return []

        # Get all .jsonl files and return names without extension
        jsonl_files = [f.stem for f in data_dir.glob("*.jsonl")]
        return sorted(jsonl_files)

    # Public interface methods (required by BaseBackend)
    def load_dataset(self, name: str) -> t.List[t.Dict[str, t.Any]]:
        """Load dataset from JSONL file."""
        return self._load("datasets", name)

    def load_experiment(self, name: str) -> t.List[t.Dict[str, t.Any]]:
        """Load experiment from JSONL file."""
        return self._load("experiments", name)

    def save_dataset(
        self,
        name: str,
        data: t.List[t.Dict[str, t.Any]],
        data_model: t.Optional[t.Type[BaseModel]] = None,
    ) -> None:
        """Save dataset to JSONL file."""
        self._save("datasets", name, data, data_model)

    def save_experiment(
        self,
        name: str,
        data: t.List[t.Dict[str, t.Any]],
        data_model: t.Optional[t.Type[BaseModel]] = None,
    ) -> None:
        """Save experiment to JSONL file."""
        self._save("experiments", name, data, data_model)

    def list_datasets(self) -> t.List[str]:
        """List all dataset names."""
        return self._list("datasets")

    def list_experiments(self) -> t.List[str]:
        """List all experiment names."""
        return self._list("experiments")

    def __repr__(self) -> str:
        return f"LocalJSONLBackend(root_dir='{self.root_dir}')"

    __str__ = __repr__
