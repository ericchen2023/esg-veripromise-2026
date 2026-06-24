"""Block until all 4 Wave C score_summary.csv files exist, printing progress every 60s."""
import time, pathlib

NAMES = ["p2ab_aug_mask10", "p2ac_aug_mix", "p2ad_rdrop05", "p2ae_msd5"]
ROOT = pathlib.Path("reports/experiments")
LOGS = pathlib.Path("outputs/logs")

def latest_epoch(name: str) -> str:
    p = LOGS / name / "seed42.jsonl"
    if not p.exists():
        return "(no log)"
    last_epoch = None
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if '"event": "epoch"' in line:
                last_epoch = line.strip()
    if not last_epoch:
        return "(no epoch yet)"
    import json
    d = json.loads(last_epoch)
    return f"fold={d['fold']} epoch={d['epoch']} score={d['scores']['final_weighted_score']:.4f}"

t0 = time.time()
while True:
    done = [n for n in NAMES if (ROOT / n / "score_summary.csv").exists()]
    print(f"[{int(time.time()-t0)}s] done {len(done)}/4")
    for n in NAMES:
        mark = "OK" if n in done else "..."
        print(f"  [{mark}] {n:25s} {latest_epoch(n)}", flush=True)
    if len(done) == 4:
        print("[wait_wave_c] ALL DONE")
        break
    if time.time() - t0 > 5400:
        print("[wait_wave_c] TIMEOUT 90min")
        break
    time.sleep(60)
