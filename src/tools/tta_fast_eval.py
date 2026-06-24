"""Fast exact evaluator for per-stem/per-view TTA ensemble searches.

The historical ``u10_per_task_tta`` search path mixed probabilities and then
converted every candidate to string labels before calling sklearn F1. That is
easy to audit, but expensive when AP-D style searches evaluate thousands of
candidate weight tuples. This module keeps the same objective while evaluating
labels as integer arrays.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from src.data.dataset import LABEL2ID, LABEL_DOMAINS, NUM_LABELS, TASKS
from src.eval.metrics import FIELD_WEIGHTS


def encode_truth(records: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Encode record labels to integer arrays using the project label order."""
    truth: dict[str, np.ndarray] = {}
    for task in TASKS:
        truth[task] = np.asarray([LABEL2ID[task][str(record[task])] for record in records], dtype=np.int16)
    return truth


def decode_label_ids(
    label_ids: dict[str, np.ndarray],
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Decode integer predictions to the same row format used by ensemble tools."""
    n_rows = len(records)
    rows: list[dict[str, Any]] = []
    for row_index in range(n_rows):
        row = {"id": records[row_index].get("id", row_index)}
        for task in TASKS:
            row[task] = LABEL_DOMAINS[task][int(label_ids[task][row_index])]
        rows.append(row)
    return rows


def apply_constraints_to_label_ids(pred_ids: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Vectorized equivalent of ``apply_constraints_batch`` for task labels."""
    out = {task: np.asarray(pred_ids[task]).copy() for task in TASKS}

    promise_no = out["promise_status"] == LABEL2ID["promise_status"]["No"]
    out["verification_timeline"][promise_no] = LABEL2ID["verification_timeline"]["N/A"]
    out["evidence_status"][promise_no] = LABEL2ID["evidence_status"]["N/A"]
    out["evidence_quality"][promise_no] = LABEL2ID["evidence_quality"]["N/A"]

    evidence_no = out["evidence_status"] == LABEL2ID["evidence_status"]["No"]
    out["evidence_quality"][evidence_no] = LABEL2ID["evidence_quality"]["N/A"]
    return out


def _f1_for_label(y_true: np.ndarray, y_pred: np.ndarray, label_id: int) -> float:
    true_pos = int(np.count_nonzero((y_true == label_id) & (y_pred == label_id)))
    false_pos = int(np.count_nonzero((y_true != label_id) & (y_pred == label_id)))
    false_neg = int(np.count_nonzero((y_true == label_id) & (y_pred != label_id)))
    denom = (2 * true_pos) + false_pos + false_neg
    if denom == 0:
        return 0.0
    return float((2 * true_pos) / denom)


def score_label_ids(
    truth_ids: dict[str, np.ndarray],
    pred_ids: dict[str, np.ndarray],
) -> dict[str, float]:
    """Exact competition score on integer label arrays."""
    out: dict[str, float] = {}
    total = 0.0
    for task, weight in FIELD_WEIGHTS.items():
        if task in ("promise_status", "evidence_status"):
            score = _f1_for_label(truth_ids[task], pred_ids[task], LABEL2ID[task]["Yes"])
        else:
            score = float(
                sum(_f1_for_label(truth_ids[task], pred_ids[task], label_id) for label_id in range(NUM_LABELS[task]))
                / NUM_LABELS[task]
            )
        out[task] = score
        total += weight * score
    out["final_weighted_score"] = float(total)
    return out


class FastTTAEvaluator:
    """Cache probability tensors and score TTA weight candidates exactly."""

    def __init__(
        self,
        *,
        per_stem_per_view: dict[str, dict[str, dict[str, np.ndarray]]],
        records: Sequence[dict[str, Any]],
        stems: Sequence[str],
        views: Sequence[str],
    ) -> None:
        self.records = list(records)
        self.stems = tuple(stems)
        self.views = tuple(views)
        self.truth_ids = encode_truth(self.records)
        self._stack: dict[str, np.ndarray] = {}

        for task in TASKS:
            view_blocks = []
            expected_shape = (len(self.records), NUM_LABELS[task])
            for view in self.views:
                stem_blocks = []
                for stem in self.stems:
                    arr = np.asarray(per_stem_per_view[stem][view][task], dtype=np.float64)
                    if arr.shape != expected_shape:
                        raise ValueError(
                            f"{stem}/{view}/{task} shape mismatch: got {arr.shape}, expected {expected_shape}"
                        )
                    stem_blocks.append(arr)
                view_blocks.append(np.stack(stem_blocks, axis=0))
            self._stack[task] = np.stack(view_blocks, axis=0)

    @staticmethod
    def _normalized(weights: Sequence[float], expected_len: int, label: str) -> np.ndarray:
        arr = np.asarray(weights, dtype=np.float64)
        if arr.shape != (expected_len,):
            raise ValueError(f"{label} expected {expected_len} weights, got {arr.shape}")
        total = float(arr.sum())
        if total <= 0.0:
            raise ValueError(f"{label} weights must have positive sum")
        return arr / total

    def mix_probs(
        self,
        *,
        stem_weights_per_task: dict[str, Sequence[float]],
        view_alpha_per_task: dict[str, Sequence[float]],
    ) -> dict[str, np.ndarray]:
        """Return final mixed probabilities for each task."""
        out: dict[str, np.ndarray] = {}
        for task in TASKS:
            stem_weights = self._normalized(stem_weights_per_task[task], len(self.stems), f"{task} stem")
            view_weights = self._normalized(view_alpha_per_task[task], len(self.views), f"{task} view")
            combined_weights = view_weights[:, None] * stem_weights[None, :]
            out[task] = np.tensordot(combined_weights, self._stack[task], axes=([0, 1], [0, 1]))
        return out

    def predict_label_ids(
        self,
        *,
        stem_weights_per_task: dict[str, Sequence[float]],
        view_alpha_per_task: dict[str, Sequence[float]],
    ) -> dict[str, np.ndarray]:
        mixed = self.mix_probs(
            stem_weights_per_task=stem_weights_per_task,
            view_alpha_per_task=view_alpha_per_task,
        )
        raw = {task: mixed[task].argmax(axis=1).astype(np.int16) for task in TASKS}
        return apply_constraints_to_label_ids(raw)

    def score(
        self,
        *,
        stem_weights_per_task: dict[str, Sequence[float]],
        view_alpha_per_task: dict[str, Sequence[float]],
    ) -> dict[str, float]:
        pred_ids = self.predict_label_ids(
            stem_weights_per_task=stem_weights_per_task,
            view_alpha_per_task=view_alpha_per_task,
        )
        return score_label_ids(self.truth_ids, pred_ids)

    def score_and_predictions(
        self,
        *,
        stem_weights_per_task: dict[str, Sequence[float]],
        view_alpha_per_task: dict[str, Sequence[float]],
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        pred_ids = self.predict_label_ids(
            stem_weights_per_task=stem_weights_per_task,
            view_alpha_per_task=view_alpha_per_task,
        )
        return score_label_ids(self.truth_ids, pred_ids), decode_label_ids(pred_ids, self.records)