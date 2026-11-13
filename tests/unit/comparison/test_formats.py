import csv
import io
import xml.etree.ElementTree as ET

import pytest

from ragas.comparison import ComparisonPlan, MetricSpec, compare
from ragas.comparison.formats import read_csv, render_csv, render_junit
from ragas.comparison.io import write_report


def report(value=1, name="score"):
    plan = ComparisonPlan(metrics=(MetricSpec(name),), mode="observed")
    return compare(
        [{"case_id": "a", name: 1}], [{"case_id": "a", name: value}], plan=plan
    )


@pytest.mark.parametrize("value,failures,errors", [(1, 0, 0), (0, 1, 0), (None, 0, 1)])
def test_junit_preserves_required_inconclusive_as_error(value, failures, errors):
    suite = ET.fromstring(render_junit(report(value)))
    assert int(suite.attrib["tests"]) == 1
    assert int(suite.attrib["failures"]) == failures
    assert int(suite.attrib["errors"]) == errors
    assert len(suite.findall("testcase")) == 1


def test_junit_escapes_xml_and_csv_neutralizes_formula_names():
    result = report(name="=SUM(A1)|<script>\x00")
    suite = ET.fromstring(render_junit(result))
    assert "<script>" in suite.find("testcase").attrib["name"]
    csv_rows = list(csv.DictReader(io.StringIO(render_csv(result))))
    assert csv_rows[0]["metric"].startswith("'=SUM")
    assert csv_rows[0]["eligible_cases"] == "1"


def test_csv_reader_preserves_ids_and_explicit_numeric_conversion(tmp_path):
    path = tmp_path / "results.csv"
    path.write_text('case_id,score,language\n"001",0.5,en\n"a,b",,de\n')
    plan = ComparisonPlan(metrics=(MetricSpec("score"),), mode="observed")
    rows = list(read_csv(path, plan=plan))
    assert rows[0]["case_id"] == "001"
    assert rows[0]["score"] == 0.5
    assert rows[1]["case_id"] == "a,b"
    assert rows[1]["score"] is None


@pytest.mark.parametrize(
    "content",
    [
        "case_id,score,score\na,1,0\n",
        "case_id,score\na,1,unexpected\n",
        "case_id,score\na,true\n",
        "case_id,score\na,9007199254740993\n",
    ],
)
def test_csv_ambiguity_and_invalid_numbers_reject(tmp_path, content):
    path = tmp_path / "results.csv"
    path.write_text(content)
    with pytest.raises(ValueError):
        list(read_csv(path, plan=ComparisonPlan(metrics=(MetricSpec("score"),))))


def test_all_file_export_formats_use_atomic_writer(tmp_path):
    result = report()
    for format_name in ("csv", "junit"):
        path = tmp_path / format_name
        write_report(path, result, output_format=format_name)
        assert path.read_text()
