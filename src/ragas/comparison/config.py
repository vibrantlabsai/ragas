"""Explicit plans and trusted resource limits for paired comparisons."""

from dataclasses import dataclass
from math import isfinite
from typing import Any, Optional, Tuple, Union

Scalar = Union[str, int, float, bool, None]


def text(value: Any, label: str) -> str:
    """Validate a field name or identifier without silently coercing it."""
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 16384:
        raise ValueError(f"{label} must be a nonempty bounded string")
    return value


def number(value: Any, label: str) -> float:
    """Convert an ordinary finite number, rejecting booleans and overflow."""
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite number, not a boolean or string")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} exceeds the numeric range") from exc
    if not isfinite(result) or abs(result) > 1e150:
        raise ValueError(f"{label} must be finite with magnitude at most 1e150")
    if type(value) is int and int(result) != value:
        raise ValueError(f"{label} cannot be represented exactly as a float")
    return result


def integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    """Validate bounded integer settings without accepting boolean flags."""
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def sequence(value: Any, label: str, maximum: int = 100000) -> tuple:
    """Copy a bounded concrete sequence, never consume an unbounded iterable."""
    if not isinstance(value, (tuple, list)) or len(value) > maximum:
        raise ValueError(f"{label} must be a bounded list or tuple")
    return tuple(value)


def scalar_key(value: Scalar) -> Tuple[str, str]:
    """An exact, type-aware key for identity and slice attributes."""
    if value is None:
        return ("null", "")
    if type(value) is bool:
        return ("bool", str(value))
    if type(value) is str:
        if len(value.encode("utf-8")) > 16384:
            raise ValueError("attribute string exceeds limit")
        return ("str", value)
    if type(value) is int:
        if value.bit_length() > 4096:
            raise ValueError("attribute integer exceeds limit")
        return ("int", str(value))
    if type(value) is float and isfinite(value):
        return ("float", value.hex())
    raise ValueError("attributes must be finite JSON scalars")


@dataclass(frozen=True)
class ComparisonLimits:
    """Caller-controlled ceilings; input files cannot increase these limits."""

    max_rows: int = 100000
    max_cases: int = 50000
    max_repeats: int = 1000
    max_metrics: int = 100
    max_slices: int = 100
    max_fields: int = 100
    max_string_bytes: int = 16384
    max_selected_bytes: int = 64 * 1024 * 1024
    max_bootstrap_work: int = 100000000
    max_details: int = 10000
    max_file_bytes: int = 128 * 1024 * 1024
    max_line_bytes: int = 1024 * 1024
    max_json_depth: int = 20
    max_json_nodes: int = 1000000
    max_alignment_cells: int = 1000000
    max_group_memberships: int = 1000000

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            integer(value, name, 1, 10**12)


@dataclass(frozen=True)
class MetricSpec:
    """One numeric metric and its tolerated degradation in original units."""

    name: str
    column: Optional[str] = None
    direction: str = "higher"
    margin: float = 0.0
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    required: bool = True

    def __post_init__(self) -> None:
        text(self.name, "metric name")
        object.__setattr__(
            self,
            "column",
            text(self.name if self.column is None else self.column, "metric column"),
        )
        if self.direction not in ("higher", "lower"):
            raise ValueError("direction must be higher or lower")
        margin = number(self.margin, "margin")
        if margin < 0:
            raise ValueError("margin must be nonnegative")
        object.__setattr__(self, "margin", margin)
        for field in ("minimum", "maximum"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, number(value, field))
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                raise ValueError("minimum must not exceed maximum")
        if type(self.required) is not bool:
            raise ValueError("required must be a boolean")

    @property
    def score_column(self) -> str:
        """The validated concrete column used to read numeric observations."""
        return self.name if self.column is None else self.column

    @property
    def orientation(self) -> int:
        """Multiply raw candidate-minus-baseline changes by this sign."""
        return 1 if self.direction == "higher" else -1


@dataclass(frozen=True)
class SliceSpec:
    """A predeclared subset of cases, selected by exact scalar membership."""

    name: str
    field: str
    values: Tuple[Scalar, ...]
    required: bool = True

    def __post_init__(self) -> None:
        text(self.name, "slice name")
        text(self.field, "slice field")
        if self.name == "overall":
            raise ValueError("overall is reserved for the full cohort")
        values = sequence(self.values, "slice values", 10000)
        if not values:
            raise ValueError("slice values cannot be empty")
        keys = [scalar_key(value) for value in values]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate slice values")
        object.__setattr__(
            self, "values", tuple(v for _, v in sorted(zip(keys, values)))
        )
        if type(self.required) is not bool:
            raise ValueError("required must be a boolean")


@dataclass(frozen=True)
class ExpectedCase:
    """One planned case, including repeat IDs and stable case attributes."""

    case_id: str
    repeats: Tuple[str, ...] = ()
    attributes: Tuple[Tuple[str, Scalar], ...] = ()

    def __post_init__(self) -> None:
        text(self.case_id, "case_id")
        repeats = sequence(self.repeats, "expected repeats", 100000)
        for repeat in repeats:
            text(repeat, "repeat_id")
        if len(set(repeats)) != len(repeats):
            raise ValueError("duplicate expected repeat_id")
        object.__setattr__(self, "repeats", tuple(sorted(repeats)))
        attributes = sequence(self.attributes, "expected attributes", 1000)
        validated = []
        for pair in attributes:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("attributes require field/value pairs")
            name, value = pair
            text(name, "attribute field")
            scalar_key(value)
            validated.append((name, value))
        if len({name for name, _ in validated}) != len(validated):
            raise ValueError("duplicate expected attribute")
        object.__setattr__(self, "attributes", tuple(sorted(validated)))


@dataclass(frozen=True)
class ComparisonPlan:
    """The fixed pairing, cohort, statistical and CI policy for a comparison.

    ``bootstrap`` makes approximate case-level inference. ``observed`` reports
    only the supplied finite benchmark. Neither mode imputes absent outcomes.
    """

    metrics: Tuple[MetricSpec, ...]
    case_id: str = "case_id"
    repeat_id: Optional[str] = None
    case_fields: Tuple[str, ...] = ()
    status_field: Optional[str] = None
    success_values: Tuple[Scalar, ...] = ("success",)
    slices: Tuple[SliceSpec, ...] = ()
    expected_cases: Optional[Tuple[ExpectedCase, ...]] = None
    mode: str = "bootstrap"
    confidence: float = 0.95
    resamples: int = 5000
    seed: int = 0
    min_cases: int = 20
    minimum_coverage: float = 1.0
    adjustment: str = "bonferroni"
    detail_limit: int = 100

    def __post_init__(self) -> None:
        metrics = sequence(self.metrics, "metrics", 1000)
        if not metrics or not all(isinstance(m, MetricSpec) for m in metrics):
            raise ValueError("metrics must contain MetricSpec objects")
        if not any(m.required for m in metrics):
            raise ValueError("at least one metric must be required")
        if len({m.name for m in metrics}) != len(metrics):
            raise ValueError("duplicate metric name")
        object.__setattr__(
            self, "metrics", tuple(sorted(metrics, key=lambda m: m.name))
        )
        text(self.case_id, "case_id field")
        for field in ("repeat_id", "status_field"):
            value = getattr(self, field)
            if value is not None:
                text(value, field)
        case_fields = sequence(self.case_fields, "case_fields", 1000)
        for field in case_fields:
            text(field, "case field")
        if len(set(case_fields)) != len(case_fields):
            raise ValueError("duplicate case field")
        object.__setattr__(self, "case_fields", tuple(sorted(case_fields)))
        success_values = sequence(self.success_values, "success_values", 1000)
        if not success_values:
            raise ValueError("success_values cannot be empty")
        for value in success_values:
            scalar_key(value)
        object.__setattr__(self, "success_values", success_values)
        slices = sequence(self.slices, "slices", 1000)
        if not all(isinstance(s, SliceSpec) for s in slices):
            raise ValueError("slices must contain SliceSpec objects")
        if len({s.name for s in slices}) != len(slices):
            raise ValueError("duplicate slice name")
        object.__setattr__(self, "slices", tuple(sorted(slices, key=lambda s: s.name)))
        if self.expected_cases is not None:
            cases = sequence(self.expected_cases, "expected_cases")
            if not cases or not all(isinstance(c, ExpectedCase) for c in cases):
                raise ValueError("expected_cases must contain ExpectedCase objects")
            if len({c.case_id for c in cases}) != len(cases):
                raise ValueError("duplicate expected case_id")
            for case in cases:
                if bool(case.repeats) != (self.repeat_id is not None):
                    raise ValueError(
                        "expected repeats must match repeat_id configuration"
                    )
                if set(dict(case.attributes)) != set(self.selected_fields):
                    raise ValueError(
                        "expected attributes must match selected case fields"
                    )
            object.__setattr__(
                self, "expected_cases", tuple(sorted(cases, key=lambda c: c.case_id))
            )
        if self.mode not in ("bootstrap", "observed"):
            raise ValueError("mode must be bootstrap or observed")
        if self.adjustment not in ("bonferroni", "none"):
            raise ValueError("adjustment must be bonferroni or none")
        confidence = number(self.confidence, "confidence")
        if not 0 < confidence < 1:
            raise ValueError("confidence must be between zero and one")
        object.__setattr__(self, "confidence", confidence)
        coverage = number(self.minimum_coverage, "minimum_coverage")
        if not 0 <= coverage <= 1:
            raise ValueError("minimum_coverage must be between zero and one")
        object.__setattr__(self, "minimum_coverage", coverage)
        integer(self.resamples, "resamples", 1, 1000000)
        integer(self.seed, "seed", 0, 2**64 - 1)
        integer(self.min_cases, "min_cases", 2, 100000)
        integer(self.detail_limit, "detail_limit", 0, 100000)

    @property
    def selected_fields(self) -> Tuple[str, ...]:
        """All fields required to establish identity and fixed slice membership."""
        return tuple(sorted(set(self.case_fields) | {s.field for s in self.slices}))

    @property
    def family_size(self) -> int:
        """Include optional and empty comparisons in the declared family."""
        return len(self.metrics) * (1 + len(self.slices))

    def check_limits(self, limits: ComparisonLimits) -> None:
        """Apply caller trust boundaries before any rows or random samples."""
        quantities = (
            (len(self.metrics), limits.max_metrics, "metrics"),
            (len(self.slices), limits.max_slices, "slices"),
            (len(self.selected_fields), limits.max_fields, "fields"),
            (len(self.expected_cases or ()), limits.max_cases, "cases"),
            (self.detail_limit, limits.max_details, "details"),
        )
        for actual, limit, label in quantities:
            if actual > limit:
                raise ValueError(f"comparison {label} limit exceeded")
        for case in self.expected_cases or ():
            if len(case.repeats) > limits.max_repeats:
                raise ValueError("expected repeats limit exceeded")
        expected_rows = sum(
            max(1, len(case.repeats)) for case in self.expected_cases or ()
        )
        if expected_rows > limits.max_rows:
            raise ValueError("expected observation rows limit exceeded")
        selected_bytes = 0
        for value in self._selected_scalars():
            kind, encoded = scalar_key(value)
            size = len(encoded.encode("utf-8"))
            if kind == "str" and size > limits.max_string_bytes:
                raise ValueError("plan selected string limit exceeded")
            selected_bytes += size + len(kind)
            if selected_bytes > limits.max_selected_bytes:
                raise ValueError("plan selected evidence byte limit exceeded")

    def _selected_scalars(self):
        """Visit bounded plan evidence without serializing a second full copy."""
        yield self.case_id
        if self.repeat_id is not None:
            yield self.repeat_id
        if self.status_field is not None:
            yield self.status_field
        yield from self.selected_fields
        yield from self.success_values
        for metric in self.metrics:
            yield metric.name
            yield metric.column
        for slice_spec in self.slices:
            yield slice_spec.name
            yield slice_spec.field
            yield from slice_spec.values
        for case in self.expected_cases or ():
            yield case.case_id
            yield from case.repeats
            for field, value in case.attributes:
                yield field
                yield value
