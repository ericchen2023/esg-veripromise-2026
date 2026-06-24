"""Run all Phase 2 ablation configs sequentially.

For each config under configs/exp_p2*.yaml:
  - run python -m src.train_kfold --config <cfg>
  - log timing
After all done:
  - run src.tools.ablate_summary

Single seed (each config already pinned to seeds:[42]).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    cfgs = sorted(Path("configs").glob("exp_p2*.yaml"))
    # skip already-completed (score_summary.csv exists)
    pending = []
    for c in cfgs:
        name = c.stem.replace("exp_", "")
        done = Path("reports/experiments") / name / "score_summary.csv"
        if done.exists():
            print(f"[skip] {c.name} (done -> {done})")
        else:
            pending.append(c)
    print(f"[ablate] {len(pending)} pending / {len(cfgs)} total")
    for c in pending:
        print(f"  - {c.name}")
    start = time.time()
    for i, cfg in enumerate(pending, 1):
        t0 = time.time()
        print(f"\n{'='*70}\n[{i}/{len(pending)}] {cfg.name}\n{'='*70}")
        rc = subprocess.call(
            [sys.executable, "-m", "src.train_kfold", "--config", str(cfg)]
        )
        dt = time.time() - t0
        print(f"[ablate] {cfg.name} done in {dt/60:.1f} min, rc={rc}")
    print(f"\n[ablate] total {time.time()-start:.1f}s")
    subprocess.call([sys.executable, "-m", "src.tools.ablate_summary"])


if __name__ == "__main__":
    main()
