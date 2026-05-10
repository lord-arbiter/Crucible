"""JSON salvage tests for critic outputs.

Critics sometimes return prose around the JSON object or slightly malformed
JSON. The salvage logic should always return a dict with the expected
keys so downstream aggregation never crashes.
"""
from __future__ import annotations

from src.critics import _extract_json_loose, _parse_error_payload


def test_clean_json_passes_through():
    raw = '{"score": 8.0, "verdict": "GOOD", "rationale": "ok", "evidence": []}'
    out = _extract_json_loose(raw)
    assert out["score"] == 8.0
    assert out["verdict"] == "GOOD"


def test_json_with_prose_around_it_recovers():
    raw = (
        "Sure — here is my JSON output:\n"
        '```json\n{"score": 4.2, "verdict": "JERKY", "rationale": "shaky", "evidence": []}\n```\n'
        "Hope this helps."
    )
    out = _extract_json_loose(raw)
    assert out["verdict"] == "JERKY"
    assert out["score"] == 4.2


def test_completely_unstructured_returns_parse_error_payload():
    out = _extract_json_loose("the model just rambled with no JSON anywhere")
    assert out["verdict"] == "PARSE_ERROR"
    assert isinstance(out["score"], float)
    assert isinstance(out["evidence"], list)


def test_empty_string_returns_parse_error():
    out = _extract_json_loose("")
    assert out["verdict"] == "PARSE_ERROR"


def test_parse_error_payload_keys():
    payload = _parse_error_payload("foo")
    assert set(payload.keys()) >= {"score", "verdict", "rationale", "evidence"}


def test_parse_error_truncates_long_raw():
    payload = _parse_error_payload("x" * 5000)
    assert len(payload["rationale"]) <= 500


def test_recovers_from_trailing_comma_via_block_grab():
    """Even when JSON is malformed, we attempt to grab a JSON-looking block."""
    raw = '{"score": 5.5, "verdict": "ACCEPTABLE", "rationale": "ok", "evidence": []}, extra'
    out = _extract_json_loose(raw)
    assert out["score"] == 5.5
