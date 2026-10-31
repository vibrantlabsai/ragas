"""Failed JSONL overwrites must preserve the last complete artifact."""

import errno
import json
import os

import pytest

from ragas import Dataset, Experiment
from ragas.backends import LocalJSONLBackend


@pytest.mark.parametrize("table_type", [Dataset, Experiment])
@pytest.mark.parametrize("exists", [False, True], ids=["new", "existing"])
def test_failed_jsonl_serialization_preserves_artifact(tmp_path, table_type, exists):
    backend = LocalJSONLBackend(str(tmp_path))
    table = table_type("results", backend, data=[{"id": "old", "score": 0.75}])
    directory = tmp_path / ("datasets" if table_type is Dataset else "experiments")
    path = directory / "results.jsonl"
    previous = None
    if exists:
        table.save()
        previous = path.read_bytes()

    replacement = table_type(
        "results",
        backend,
        data=[{"id": "new"}, {"not_json_serializable": object()}],
    )
    with pytest.raises(TypeError):
        replacement.save()

    if exists:
        assert path.read_bytes() == previous
        assert [json.loads(line) for line in path.read_text().splitlines()] == [
            {"id": "old", "score": 0.75}
        ]
        assert list(table_type.load("results", backend)) == list(table)
    else:
        assert not path.exists()
    assert sorted(p.name for p in directory.iterdir()) == (
        ["results.jsonl"] if exists else []
    )


@pytest.mark.parametrize("table_type", [Dataset, Experiment])
@pytest.mark.parametrize("exists", [False, True], ids=["new", "existing"])
def test_failed_jsonl_commit_preserves_artifact(
    tmp_path, monkeypatch, table_type, exists
):
    backend = LocalJSONLBackend(str(tmp_path))
    table = table_type("results", backend, data=[{"id": "old"}])
    directory = tmp_path / ("datasets" if table_type is Dataset else "experiments")
    path = directory / "results.jsonl"
    if exists:
        table.save()
    previous = path.read_bytes() if exists else None
    failure = OSError(errno.ENOSPC, "simulated full storage during commit")

    def fail_replace(source, destination):
        assert (path.read_bytes() if path.exists() else None) == previous
        raise failure

    monkeypatch.setattr(os, "replace", fail_replace)
    replacement = table_type("results", backend, data=[{"id": "new"}])
    with pytest.raises(OSError) as raised:
        replacement.save()
    assert raised.value is failure
    assert (path.read_bytes() if path.exists() else None) == previous
    assert sorted(p.name for p in directory.iterdir()) == (
        ["results.jsonl"] if exists else []
    )


@pytest.mark.parametrize("table_type", [Dataset, Experiment])
@pytest.mark.parametrize(
    "rows", [[], [{"text": "雪", "scores": [1, 0.25], "ok": True}]]
)
def test_jsonl_complete_replacement_round_trip(tmp_path, table_type, rows):
    backend = LocalJSONLBackend(str(tmp_path))
    table_type("results", backend, data=[{"id": "old"}]).save()
    replacement = table_type("results", backend, data=rows)
    replacement.save()
    assert list(table_type.load("results", backend)) == rows
    directory = tmp_path / ("datasets" if table_type is Dataset else "experiments")
    assert sorted(p.name for p in directory.iterdir()) == ["results.jsonl"]
    assert [
        json.loads(line)
        for line in (directory / "results.jsonl").read_text().splitlines()
    ] == rows


@pytest.mark.parametrize("table_type", [Dataset, Experiment])
@pytest.mark.parametrize("exists", [False, True], ids=["new", "existing"])
def test_jsonl_write_failure_preserves_artifact(
    tmp_path, monkeypatch, table_type, exists
):
    from ragas.backends import local_jsonl

    backend = LocalJSONLBackend(str(tmp_path))
    table = table_type("results", backend, data=[{"id": "old"}])
    directory = tmp_path / ("datasets" if table_type is Dataset else "experiments")
    path = directory / "results.jsonl"
    if exists:
        table.save()
    previous = path.read_bytes() if exists else None
    failure = OSError(errno.ENOSPC, "simulated full storage during second row")
    original_open = open

    class FailingWriter:
        def __init__(self, stream):
            self.stream = stream
            self.writes = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def write(self, value):
            self.writes += 1
            if self.writes == 2:
                raise failure
            return self.stream.write(value)

    def failing_open(path, mode, *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        return FailingWriter(stream) if mode in ("w", "x") else stream

    monkeypatch.setattr(local_jsonl, "open", failing_open, raising=False)
    replacement = table_type("results", backend, data=[{"id": "new"}, {"id": "second"}])
    with pytest.raises(OSError) as raised:
        replacement.save()
    assert raised.value is failure
    assert (path.read_bytes() if path.exists() else None) == previous
    assert sorted(p.name for p in directory.iterdir()) == (
        ["results.jsonl"] if exists else []
    )


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
@pytest.mark.parametrize("table_type", [Dataset, Experiment])
def test_jsonl_overwrite_preserves_permissions(tmp_path, table_type):
    import stat

    backend = LocalJSONLBackend(str(tmp_path))
    table = table_type("results", backend, data=[{"id": "old"}])
    table.save()
    directory = tmp_path / ("datasets" if table_type is Dataset else "experiments")
    path = directory / "results.jsonl"
    path.chmod(0o640)
    table_type("results", backend, data=[]).save()
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    assert path.read_bytes() == b""


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_new_jsonl_file_uses_normal_creation_permissions(tmp_path):
    import stat

    reference = tmp_path / "reference"
    reference.write_text("reference", encoding="utf-8")
    backend = LocalJSONLBackend(str(tmp_path))
    Dataset("results", backend, data=[{"id": "new"}]).save()
    path = tmp_path / "datasets" / "results.jsonl"
    assert stat.S_IMODE(path.stat().st_mode) == stat.S_IMODE(reference.stat().st_mode)


@pytest.mark.skipif(
    os.name != "posix" or os.geteuid() == 0,
    reason="Requires POSIX write denial for a non-root user",
)
@pytest.mark.parametrize("table_type", [Dataset, Experiment])
@pytest.mark.parametrize("rows", [[], [{"id": "new"}]])
def test_jsonl_overwrite_respects_write_denial(tmp_path, monkeypatch, table_type, rows):
    backend = LocalJSONLBackend(str(tmp_path))
    table_type("results", backend, data=[{"id": "old"}]).save()
    directory = tmp_path / ("datasets" if table_type is Dataset else "experiments")
    path = directory / "results.jsonl"
    previous = path.read_bytes()
    path.chmod(0o444)
    original_open = os.open
    failures = []

    def observe_open(file, flags, *args, **kwargs):
        try:
            return original_open(file, flags, *args, **kwargs)
        except PermissionError as error:
            failures.append(error)
            raise

    monkeypatch.setattr(os, "open", observe_open)
    try:
        with pytest.raises(PermissionError) as raised:
            table_type("results", backend, data=rows).save()
        assert raised.value.errno == errno.EACCES
        if failures:
            assert raised.value is failures[0]
        assert path.read_bytes() == previous
        assert [json.loads(line) for line in path.read_text().splitlines()] == [
            {"id": "old"}
        ]
        assert sorted(p.name for p in directory.iterdir()) == ["results.jsonl"]
    finally:
        path.chmod(0o644)


@pytest.mark.parametrize("table_type", [Dataset, Experiment])
@pytest.mark.parametrize("exists", [False, True], ids=["new", "existing"])
def test_jsonl_preflight_failure_preserves_artifact(
    tmp_path, monkeypatch, table_type, exists
):
    backend = LocalJSONLBackend(str(tmp_path))
    directory = tmp_path / ("datasets" if table_type is Dataset else "experiments")
    path = directory / "results.jsonl"
    if exists:
        table_type("results", backend, data=[{"id": "old"}]).save()
    previous = path.read_bytes() if exists else None
    failure = PermissionError(errno.EACCES, "write authorization denied", str(path))

    def deny_write(file, flags):
        assert file == path
        assert flags == os.O_WRONLY
        assert path.read_bytes() == previous
        assert sorted(p.name for p in directory.iterdir()) == ["results.jsonl"]
        raise failure

    monkeypatch.setattr(os, "open", deny_write)
    replacement = table_type("results", backend, data=[{"id": "new"}])
    if exists:
        with pytest.raises(PermissionError) as raised:
            replacement.save()
        assert raised.value is failure
        assert path.read_bytes() == previous
    else:
        replacement.save()
        assert [json.loads(line) for line in path.read_text().splitlines()] == [
            {"id": "new"}
        ]
    assert sorted(p.name for p in directory.iterdir()) == ["results.jsonl"]
