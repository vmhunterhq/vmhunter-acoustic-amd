"""Tune the live decision policy on val and report it on test.

Uses the per-decision-point probabilities saved by train.py (probs_{split}.npy,
P x N x C over PREFIXES). Policy, evaluated at each decision point t:
  - p(human) >= thr_h and t >= t_h  -> HUMAN at t
  - max non-human p >= thr_m        -> that cause at t (MACHINE)
  - at the last point, nothing passed  -> UNKNOWN (MACHINE)
AMDSTATUS is HUMAN only for cause HUMAN (user rule).

Prints, for each allowed live-person-lost rate, the val-tuned setting with the fewest
recordings sent to agents, and how it does on test.
"""
import argparse, itertools, json, os
import numpy as np

PREFIXES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
CLASSES = ["human", "machine_speech", "screening", "silence", "tone"]
CAUSE = {"human": "HUMAN", "machine_speech": "MACHINE", "screening": "SCREENING", "silence": "SILENCE", "tone": "TONE"}
H = 0


def run_policy(probs, thr_h, thr_m, t_h):
    """Returns (cause index or -1 for UNKNOWN, decision time) per clip."""
    P, N, C = probs.shape
    cause = np.full(N, -2)
    when = np.full(N, PREFIXES[-1])
    for k, t in enumerate(PREFIXES):
        open_ = cause == -2
        if not open_.any():
            break
        p = probs[k]
        hum = open_ & (p[:, H] >= thr_h) & (t >= t_h)
        nonh = p[:, 1:]
        mach = open_ & ~hum & (nonh.max(1) >= thr_m)
        cause[hum] = H
        cause[mach] = 1 + nonh[mach].argmax(1)
        when[hum | mach] = t
    cause[cause == -2] = -1
    return cause, when


def score(cause, when, y):
    is_h = y == H
    said_h = cause == H
    nonh = ~is_h
    correct_cause = (cause == y)
    return {
        "human_lost": float((is_h & ~said_h).sum() / max(is_h.sum(), 1)),
        "machine_to_human": float((nonh & said_h).sum() / max(nonh.sum(), 1)),
        "unknown_rate": float((cause == -1).mean()),
        "cause_acc_nonhuman": float(correct_cause[nonh].mean()),
        "mean_ms": float(when.mean() * 1000),
        "human_mean_ms": float(when[is_h & said_h].mean() * 1000) if (is_h & said_h).any() else float("nan"),
        "machine_mean_ms": float(when[nonh].mean() * 1000),
        "pct_decided_by_1s": float((when <= 1.0).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/stream_v1")
    a = ap.parse_args()
    pv, yv = np.load(f"{a.run}/probs_val.npy"), np.load(f"{a.run}/labels_val.npy")
    pt, yt = np.load(f"{a.run}/probs_test.npy"), np.load(f"{a.run}/labels_test.npy")

    grid = list(itertools.product(
        [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97],  # thr_h
        [0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995],             # thr_m
        [1.0, 1.25, 1.5, 1.75, 2.0]))                          # earliest HUMAN decision
    rows = []
    for thr_h, thr_m, t_h in grid:
        s = score(*run_policy(pv, thr_h, thr_m, t_h), yv)
        rows.append(((thr_h, thr_m, t_h), s))

    print("argmax at 2.0 s, for reference (test):")
    p2 = pt[-1]
    print(" ", json.dumps({k: round(v, 4) for k, v in score(p2.argmax(1), np.full(len(yt), 2.0), yt).items()}))

    chosen = {}
    print("\nbudget = max live-person-lost on val; pick fewest recordings->agent, then fastest")
    print(f"{'budget':>6} | {'thr_h':>5} {'thr_m':>5} {'t_h':>4} | {'split':5} {'lost':>6} {'m->h':>6} {'unk':>6} "
          f"{'cause':>6} {'mean ms':>7} {'mach ms':>7} {'hum ms':>7} {'<=1s':>5}")
    for budget in (0.01, 0.015, 0.02, 0.03, 0.05):
        ok = [r for r in rows if r[1]["human_lost"] <= budget]
        if not ok:
            print(f"{budget:6.1%} | no setting meets this on val")
            continue
        cfg, sv = min(ok, key=lambda r: (round(r[1]["machine_to_human"], 4), r[1]["mean_ms"]))
        st = score(*run_policy(pt, *cfg), yt)
        chosen[str(budget)] = {"thr_h": cfg[0], "thr_m": cfg[1], "t_h": cfg[2], "val": sv, "test": st}
        for name, s in (("val", sv), ("test", st)):
            print(f"{budget:6.1%} | {cfg[0]:5.2f} {cfg[1]:5.3f} {cfg[2]:4.2f} | {name:5} {s['human_lost']:6.2%} "
                  f"{s['machine_to_human']:6.2%} {s['unknown_rate']:6.2%} {s['cause_acc_nonhuman']:6.2%} "
                  f"{s['mean_ms']:7.0f} {s['machine_mean_ms']:7.0f} {s['human_mean_ms']:7.0f} {s['pct_decided_by_1s']:5.0%}")
    json.dump(chosen, open(f"{a.run}/early_exit.json", "w"), indent=1)


if __name__ == "__main__":
    main()
