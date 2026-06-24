"""Hierarchical post-processing enforcing label dependency constraints.

Rules (derived from competition spec):
  - promise_status == "No"  =>  verification_timeline = evidence_status = evidence_quality = "N/A"
                                promise_string = ""
                                evidence_string = ""
  - evidence_status == "No" =>  evidence_quality = "N/A"
                                evidence_string = ""
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any


def apply_constraints(record: dict[str, Any]) -> dict[str, Any]:
    """Return a new record with hierarchical constraints enforced."""
    r = deepcopy(record)

    if r.get("promise_status") == "No":
        r["verification_timeline"] = "N/A"
        r["evidence_status"] = "N/A"
        r["evidence_quality"] = "N/A"
        if "promise_string" in r:
            r["promise_string"] = ""
        if "evidence_string" in r:
            r["evidence_string"] = ""

    if r.get("evidence_status") == "No":
        r["evidence_quality"] = "N/A"
        if "evidence_string" in r:
            r["evidence_string"] = ""

    return r


def apply_constraints_batch(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [apply_constraints(r) for r in records]
