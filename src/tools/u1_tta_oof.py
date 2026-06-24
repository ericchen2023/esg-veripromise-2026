"""U1 TTA OOF evaluation for the Phase 14 v12 ensemble.

This script keeps the training data fixed and re-runs validation-fold
checkpoints with alternate token windows. It then averages those probabilities
with stored OOF probabilities and scores the existing v12 joint-hillclimb
weights after hierarchical post-processing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from src.config import load_config
from src.data.dataset import LABEL_DOMAINS, NUM_LABELS, TASKS
from src.data.loader import load_dataset
from src.data.splits import load_folds
from src.data.text_augment import get_added_tokens, get_text_transform
from src.eval.metrics import weighted_score
from src.inference.post_process import apply_constraints_batch
from src.models.multitask import MultiTaskClassifier
from src.tools.joint_hillclimb_v12 import _build_pool_v12, _load_member


DEFAULT_META = Path("reports/analysis/_ensemble/joint_hillclimb_v12_meta.json")
DEFAULT_DATA = Path("data/raw/vpesg4k_train_1000 V1.csv")
OUTPUT_DIR = Path("reports/analysis/_ensemble")


class TTADataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        global_indices: list[int],
        tokenizer,
        max_length: int,
        view: str,
        text_transform=None,
    ) -> None:
        self.records = records
        self.global_indices = global_indices
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.view = view
        self.text_transform = text_transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        text = str(self.records[index].get("data", "") or "")
        if self.text_transform is not None:
            text = self.text_transform(text)
        encoded = encode_view(self.tokenizer, text, self.max_length, self.view)
        encoded["_index"] = torch.tensor(self.global_indices[index], dtype=torch.long)
        return encoded


def tta_collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "_index": torch.stack([item["_index"] for item in batch]),
    }


def encode_view(tokenizer, text: str, max_length: int, view: str) -> dict[str, torch.Tensor]:
    if view == "head":
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

    token_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    window_size = max(1, max_length - tokenizer.num_special_tokens_to_add(pair=False))
    if len(token_ids) > window_size:
        if view == "tail":
            token_ids = token_ids[-window_size:]
        elif view == "middle":
            start_index = max(0, (len(token_ids) - window_size) // 2)
            token_ids = token_ids[start_index : start_index + window_size]
        else:
            raise ValueError(f"Unsupported TTA view: {view}")

    input_ids: list[int] = []
    if tokenizer.cls_token_id is not None:
        input_ids.append(int(tokenizer.cls_token_id))
    input_ids.extend(int(token_id) for token_id in token_ids)
    if tokenizer.sep_token_id is not None:
        input_ids.append(int(tokenizer.sep_token_id))
    attention_mask = [1] * len(input_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    pad_count = max_length - len(input_ids)
    if pad_count < 0:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
    else:
        input_ids.extend([int(pad_id)] * pad_count)
        attention_mask.extend([0] * pad_count)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def load_meta(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def active_members(meta: dict[str, Any]) -> list[str]:
    members = list(meta["members"])
    active: list[str] = []
    for member_index, member_name in enumerate(members):
        has_weight = any(float(meta["best_w"][task][member_index]) != 0.0 for task in TASKS)
        if has_weight:
            active.append(member_name)
    return active


def resolve_members(requested: str, meta: dict[str, Any]) -> list[str]:
    if requested == "all":
        return list(meta["members"])
    if requested == "active":
        return active_members(meta)
    requested_members = [part.strip() for part in requested.split(",") if part.strip()]
    missing = sorted(set(requested_members) - set(meta["members"]))
    if missing:
        raise ValueError(f"Requested members are not in v12 meta: {missing}")
    return requested_members


def config_for_stem(stem: str) -> dict[str, Any]:
    config_path = Path("configs") / f"exp_{stem}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config found for checkpoint stem: {stem} ({config_path})")
    return load_config(config_path)


def create_tokenizer(cfg: dict[str, Any]):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["backbone"])
    text_transform_name = (cfg.get("data", {}) or {}).get("text_transform")
    extra_tokens = get_added_tokens(text_transform_name)
    if extra_tokens:
        tokenizer.add_tokens(extra_tokens, special_tokens=False)
    return tokenizer, get_text_transform(text_transform_name), extra_tokens


def create_model(cfg: dict[str, Any], tokenizer, extra_tokens: list[str], checkpoint_path: Path, device: torch.device):
    model = MultiTaskClassifier(
        backbone=cfg["model"]["backbone"],
        num_labels=NUM_LABELS,
        pooling=cfg["model"].get("pooling", "cls_mean"),
        dropout=float(cfg["model"].get("dropout", 0.1)),
        msd_k=int(cfg["model"].get("msd_k", 1)),
    )
    if extra_tokens:
        model.encoder.resize_token_embeddings(len(tokenizer))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def member_seed_list(stem: str, mode: str, explicit_seeds) -> list[int]:
    if mode == "single":
        return list(explicit_seeds)
    seed_root = Path("outputs/checkpoints") / stem
    return sorted(int(path.name.replace("seed", "")) for path in seed_root.iterdir() if path.name.startswith("seed"))


def predict_one_view(
    *,
    records: list[dict[str, Any]],
    member_name: str,
    loader_spec: tuple,
    view: str,
    device: torch.device,
    use_amp: bool,
    batch_size_override: int | None,
) -> dict[str, np.ndarray]:
    mode, stem, explicit_seeds = loader_spec
    cfg = config_for_stem(stem)
    tokenizer, text_transform, extra_tokens = create_tokenizer(cfg)
    max_length = int(cfg["model"]["max_length"])
    batch_size = batch_size_override or int(cfg["training"].get("batch_size", 4)) * 2
    checkpoint_root = Path("outputs/checkpoints") / stem
    split_root = Path("data/splits") / stem
    seed_values = member_seed_list(stem, mode, explicit_seeds)
    seed_probs: list[dict[str, np.ndarray]] = []

    print(f"[tta] {member_name}: stem={stem} seeds={seed_values} view={view} max_len={max_length}")
    for seed_value in seed_values:
        fold_probs = {task: np.zeros((len(records), NUM_LABELS[task]), dtype=np.float64) for task in TASKS}
        seen = np.zeros(len(records), dtype=bool)
        folds = load_folds(split_root / f"seed{seed_value}.json")
        for fold_index, (_, val_indices) in enumerate(folds):
            checkpoint_path = checkpoint_root / f"seed{seed_value}" / f"fold{fold_index}" / "best.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
            val_index_list = [int(index_value) for index_value in val_indices]
            val_records = [records[index_value] for index_value in val_index_list]
            dataset = TTADataset(
                val_records,
                val_index_list,
                tokenizer,
                max_length=max_length,
                view=view,
                text_transform=text_transform,
            )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=tta_collate)
            model = create_model(cfg, tokenizer, extra_tokens, checkpoint_path, device)
            predicted = predict_loader(model, loader, device, use_amp)
            for task in TASKS:
                fold_probs[task][predicted["indices"]] = predicted[task]
            seen[val_index_list] = True
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if not seen.all():
            missing_count = int((~seen).sum())
            raise RuntimeError(f"{member_name}/seed{seed_value} missing {missing_count} OOF samples")
        seed_probs.append(fold_probs)

    averaged = {task: np.zeros((len(records), NUM_LABELS[task]), dtype=np.float64) for task in TASKS}
    for probs in seed_probs:
        for task in TASKS:
            averaged[task] += probs[task]
    for task in TASKS:
        averaged[task] /= len(seed_probs)
    return averaged


def predict_loader(model, loader: DataLoader, device: torch.device, use_amp: bool) -> dict[str, np.ndarray]:
    buffers: dict[str, list[np.ndarray]] = {task: [] for task in TASKS}
    indices: list[int] = []
    autocast_context = torch.amp.autocast(device_type=device.type, enabled=use_amp)
    with torch.no_grad(), autocast_context:
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            logits = model(input_ids, attention_mask)
            for task in TASKS:
                buffers[task].append(torch.softmax(logits[task].float(), dim=-1).cpu().numpy())
            indices.extend(batch["_index"].cpu().tolist())
    order = np.argsort(indices)
    output: dict[str, np.ndarray] = {"indices": np.asarray(indices, dtype=np.int32)[order]}
    for task in TASKS:
        output[task] = np.concatenate(buffers[task], axis=0)[order]
    return output


def combine_member_views(view_probs: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    combined = {task: np.zeros_like(view_probs[0][task], dtype=np.float64) for task in TASKS}
    for probs in view_probs:
        for task in TASKS:
            combined[task] += probs[task]
    for task in TASKS:
        combined[task] /= len(view_probs)
    return combined


def combine_v12(meta: dict[str, Any], per_member_probs: dict[str, dict[str, np.ndarray]], n_records: int) -> dict[str, np.ndarray]:
    members = list(meta["members"])
    accum = {task: np.zeros((n_records, NUM_LABELS[task]), dtype=np.float64) for task in TASKS}
    for task in TASKS:
        weights = [float(weight) for weight in meta["best_w"][task]]
        total_weight = sum(weights)
        if total_weight == 0:
            raise RuntimeError(f"v12 task has zero total weight: {task}")
        for member_name, weight in zip(members, weights):
            if weight == 0.0:
                continue
            accum[task] += weight * per_member_probs[member_name][task]
        accum[task] /= total_weight
    return accum


def score_probs(records: list[dict[str, Any]], probs: dict[str, np.ndarray]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        row = {"id": record.get("id", record_index)}
        for task in TASKS:
            row[task] = LABEL_DOMAINS[task][int(probs[task][record_index].argmax())]
        predictions.append(row)
    constrained = apply_constraints_batch(predictions)
    pred_dict = {task: [row[task] for row in constrained] for task in TASKS}
    truth = {task: [record[task] for record in records] for task in TASKS}
    return weighted_score(truth, pred_dict), constrained


def write_outputs(tag: str, summary_rows: list[dict[str, Any]], predictions: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / f"u1_tta_v12_{tag}_summary.csv"
    pred_path = OUTPUT_DIR / f"u1_tta_v12_{tag}_preds.csv"
    meta_path = OUTPUT_DIR / f"u1_tta_v12_{tag}_meta.json"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(predictions).to_csv(pred_path, index=False)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[wrote] {summary_path}")
    print(f"[wrote] {pred_path}")
    print(f"[wrote] {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", default=str(DEFAULT_META))
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--views", nargs="+", default=["stored", "tail"], choices=["stored", "head", "middle", "tail"])
    parser.add_argument("--members", default="active", help="active, all, or comma-separated v12 members")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    meta = load_meta(Path(args.meta))
    records, _ = load_dataset(args.data)
    selected_members = resolve_members(args.members, meta)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    use_amp = bool(device.type == "cuda" and not args.no_amp)
    print(f"[u1] N={len(records)} device={device} views={args.views} members={len(selected_members)}/{len(meta['members'])}")

    exps, loaders, _ = _build_pool_v12()
    loader_map = {member_name: loaders[member_name] for member_name in exps}

    stored_member_probs = {member_name: _load_member(member_name, len(records), loaders) for member_name in meta["members"]}
    baseline_probs = combine_v12(meta, stored_member_probs, len(records))
    baseline_score, _ = score_probs(records, baseline_probs)
    print(f"[baseline stored] joint={baseline_score['final_weighted_score']:.10f}")

    tta_member_probs = dict(stored_member_probs)
    for member_name in selected_members:
        view_buffers: list[dict[str, np.ndarray]] = []
        if "stored" in args.views:
            view_buffers.append(stored_member_probs[member_name])
        for view_name in args.views:
            if view_name == "stored":
                continue
            view_buffers.append(
                predict_one_view(
                    records=records,
                    member_name=member_name,
                    loader_spec=loader_map[member_name],
                    view=view_name,
                    device=device,
                    use_amp=use_amp,
                    batch_size_override=args.batch_size,
                )
            )
        tta_member_probs[member_name] = combine_member_views(view_buffers)

    tta_probs = combine_v12(meta, tta_member_probs, len(records))
    tta_score, tta_predictions = score_probs(records, tta_probs)
    delta = tta_score["final_weighted_score"] - baseline_score["final_weighted_score"]
    print(f"[tta] joint={tta_score['final_weighted_score']:.10f} delta={delta:+.10f}")
    print(
        "[tta tasks] "
        + " ".join(f"{task}={tta_score[task]:.6f}" for task in TASKS)
    )

    view_tag = "_".join(args.views)
    member_tag = "active" if args.members == "active" else "custom"
    tag = f"{member_tag}_{view_tag}"
    summary_rows = [
        {"variant": "stored_v12", **{task: baseline_score[task] for task in TASKS}, "weighted_score": baseline_score["final_weighted_score"], "delta_vs_stored": 0.0},
        {"variant": f"tta_{view_tag}", **{task: tta_score[task] for task in TASKS}, "weighted_score": tta_score["final_weighted_score"], "delta_vs_stored": delta},
    ]
    output_meta = {
        "views": args.views,
        "members_mode": args.members,
        "selected_members": selected_members,
        "baseline_score": baseline_score,
        "tta_score": tta_score,
        "delta_vs_stored": delta,
        "source_meta": str(args.meta),
        "data": str(args.data),
        "device": str(device),
        "use_amp": use_amp,
    }
    write_outputs(tag, summary_rows, tta_predictions, output_meta)


if __name__ == "__main__":
    main()