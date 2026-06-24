from __future__ import annotations

import numpy as np
import pytest

from src.data.dataset import NUM_LABELS, TASKS
from src.eval.metrics import weighted_score
from src.inference.post_process import apply_constraints_batch
from src.tools.tta_fast_eval import FastTTAEvaluator, encode_truth, score_label_ids
from src.tools.u10_per_task_tta import _eval_full, _eval_full_reference, random_refine_search


def _records() -> list[dict[str, object]]:
    return [
        {
            "id": 1,
            "promise_status": "Yes",
            "verification_timeline": "already",
            "evidence_status": "Yes",
            "evidence_quality": "Clear",
        },
        {
            "id": 2,
            "promise_status": "No",
            "verification_timeline": "N/A",
            "evidence_status": "N/A",
            "evidence_quality": "N/A",
        },
        {
            "id": 3,
            "promise_status": "Yes",
            "verification_timeline": "within_2_years",
            "evidence_status": "No",
            "evidence_quality": "N/A",
        },
        {
            "id": 4,
            "promise_status": "Yes",
            "verification_timeline": "between_2_and_5_years",
            "evidence_status": "Yes",
            "evidence_quality": "Misleading",
        },
        {
            "id": 5,
            "promise_status": "Yes",
            "verification_timeline": "longer_than_5_years",
            "evidence_status": "Yes",
            "evidence_quality": "Not Clear",
        },
    ]


def _probabilities(stems: tuple[str, ...], views: tuple[str, ...], n_rows: int):
    rng = np.random.default_rng(20260525)
    out = {stem: {} for stem in stems}
    for stem in stems:
        for view in views:
            out[stem][view] = {}
            for task in TASKS:
                arr = rng.random((n_rows, NUM_LABELS[task]))
                arr = arr / arr.sum(axis=1, keepdims=True)
                out[stem][view][task] = arr
    return out


def test_fast_eval_matches_reference_path(monkeypatch) -> None:
    import src.tools.u10_per_task_tta as tta

    records = _records()
    stems = ("stem_a", "stem_b")
    monkeypatch.setattr(tta, "U10_STEMS", stems)
    per_stem_per_view = _probabilities(stems, tta.VIEWS, len(records))
    stem_weights = {task: (0.25, 0.75) for task in TASKS}
    view_alpha = {task: (0.2, 0.3, 0.5) for task in TASKS}

    ref_score, ref_preds = _eval_full_reference(
        per_stem_per_view=per_stem_per_view,
        stem_weights_per_task=stem_weights,
        view_alpha_per_task=view_alpha,
        records=records,
    )
    evaluator = FastTTAEvaluator(
        per_stem_per_view=per_stem_per_view,
        records=records,
        stems=stems,
        views=tta.VIEWS,
    )
    fast_score, fast_preds = _eval_full(
        per_stem_per_view=per_stem_per_view,
        stem_weights_per_task=stem_weights,
        view_alpha_per_task=view_alpha,
        records=records,
        evaluator=evaluator,
    )

    assert fast_preds == ref_preds
    for key, value in ref_score.items():
        assert fast_score[key] == pytest.approx(value, rel=1e-12, abs=1e-12)


def test_integer_score_matches_sklearn_score() -> None:
    truth_rows = _records()
    raw_pred_rows = [
        {
            "id": 1,
            "promise_status": "Yes",
            "verification_timeline": "already",
            "evidence_status": "Yes",
            "evidence_quality": "Clear",
        },
        {
            "id": 2,
            "promise_status": "No",
            "verification_timeline": "already",
            "evidence_status": "Yes",
            "evidence_quality": "Clear",
        },
        {
            "id": 3,
            "promise_status": "Yes",
            "verification_timeline": "within_2_years",
            "evidence_status": "No",
            "evidence_quality": "Clear",
        },
        {
            "id": 4,
            "promise_status": "Yes",
            "verification_timeline": "between_2_and_5_years",
            "evidence_status": "Yes",
            "evidence_quality": "Not Clear",
        },
        {
            "id": 5,
            "promise_status": "Yes",
            "verification_timeline": "already",
            "evidence_status": "Yes",
            "evidence_quality": "N/A",
        },
    ]
    pred_rows = apply_constraints_batch(raw_pred_rows)
    truth_ids = encode_truth(truth_rows)
    pred_ids = encode_truth(pred_rows)
    fast = score_label_ids(truth_ids, pred_ids)
    slow = weighted_score(
        {task: [row[task] for row in truth_rows] for task in TASKS},
        {task: [row[task] for row in pred_rows] for task in TASKS},
    )

    for key, value in slow.items():
        assert fast[key] == pytest.approx(value, rel=1e-12, abs=1e-12)


def test_random_refine_never_accepts_regression(monkeypatch) -> None:
    import src.tools.u10_per_task_tta as tta

    records = _records()
    stems = ("stem_a", "stem_b")
    monkeypatch.setattr(tta, "U10_STEMS", stems)
    per_stem_per_view = _probabilities(stems, tta.VIEWS, len(records))
    evaluator = FastTTAEvaluator(
        per_stem_per_view=per_stem_per_view,
        records=records,
        stems=stems,
        views=tta.VIEWS,
    )
    init_stem = {task: (0.5, 0.5) for task in TASKS}
    init_alpha = {task: (1.0, 0.0, 0.0) for task in TASKS}
    initial = evaluator.score(stem_weights_per_task=init_stem, view_alpha_per_task=init_alpha)

    stem, alpha, final, _ = random_refine_search(
        evaluator=evaluator,
        init_stem=init_stem,
        init_alpha=init_alpha,
        n_iters=25,
        step=0.5,
        seed=20260525,
    )

    assert final["final_weighted_score"] >= initial["final_weighted_score"]
    for task in TASKS:
        assert sum(stem[task]) == pytest.approx(1.0)
        assert sum(alpha[task]) == pytest.approx(1.0)