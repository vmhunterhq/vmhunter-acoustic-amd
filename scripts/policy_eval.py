"""Tune the live two-model decision policy on val; report on test.

  early model (trained on cut-off audio): before 2.0 s, return a non-human cause
      as soon as its probability >= thr_m (only clear recordings / silence exit early)
  final model (full 2.0 s window) at 2.0 s:
      p(human) >= thr_h            -> HUMAN
      else best non-human cause    -> that cause, or UNKNOWN if its p < 0.5
AMDSTATUS = HUMAN only for cause HUMAN.
"""
import argparse, itertools, json
import numpy as np

PREFIXES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
CLASSES = ["human", "machine_speech", "screening", "silence", "tone"]
H, UNKNOWN = 0, -1


def policy(pe, pf, thr_h, thr_m, t_min):
    N = pe.shape[1]
    cause = np.full(N, -2)
    when = np.full(N, 2.0)
    for k, t in enumerate(PREFIXES[:-1]):
        if t < t_min:
            continue
        nonh = pe[k][:, 1:]
        go = (cause == -2) & (nonh.max(1) >= thr_m)
        cause[go] = 1 + nonh[go].argmax(1)
        when[go] = t
    p = pf[-1]
    rest = cause == -2
    hum = rest & (p[:, H] >= thr_h)
    cause[hum] = H
    other = rest & ~hum
    best = 1 + p[:, 1:].argmax(1)
    conf = p[:, 1:].max(1)
    cause[other] = np.where(conf[other] >= 0.5, best[other], UNKNOWN)
    return cause, when


def score(cause, when, y):
    is_h, said_h = y == H, cause == H
    nonh = ~is_h
    return {
        "human_lost": float((is_h & ~said_h).sum() / is_h.sum()),
        "machine_to_human": float((nonh & said_h).sum() / nonh.sum()),
        "unknown": float((cause == UNKNOWN).mean()),
        "cause_acc": float((cause == y)[nonh].mean()),
        "mach_ms": float(when[nonh].mean() * 1000),
        "early_share": float((when < 2.0).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--early", default="runs/stream_v1")
    ap.add_argument("--final", default="runs/baseline_relabel")
    a = ap.parse_args()
    d = {s: (np.load(f"{a.early}/probs_{s}.npy"), np.load(f"{a.final}/probs_{s}.npy"), np.load(f"{a.final}/labels_{s}.npy"))
         for s in ("val", "test")}
    for s in d:
        assert (np.load(f"{a.early}/labels_{s}.npy") == d[s][2]).all(), "row order differs between runs"

    grid = list(itertools.product([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                  [0.95, 0.98, 0.99, 0.995, 0.998, 1.01],  # 1.01 = no early exit
                                  [0.5, 1.0, 1.25, 1.5]))
    rows = [(g, score(*policy(d["val"][0], d["val"][1], *g), d["val"][2])) for g in grid]
    out = {}
    print(f"{'budget':>6} | {'thr_h':>5} {'thr_m':>5} {'t_min':>5} | {'split':5} {'lost':>6} {'m->h':>6} "
          f"{'unk':>6} {'cause':>6} {'mach ms':>7} {'early':>6}")
    for budget in (0.01, 0.015, 0.02, 0.025, 0.03):
        ok = [r for r in rows if r[1]["human_lost"] <= budget]
        if not ok:
            print(f"{budget:6.1%} | none"); continue
        g, sv = min(ok, key=lambda r: (round(r[1]["machine_to_human"], 4), r[1]["mach_ms"]))
        st = score(*policy(d["test"][0], d["test"][1], *g), d["test"][2])
        out[str(budget)] = {"thr_h": g[0], "thr_m": g[1], "t_min": g[2], "val": sv, "test": st}
        for name, s in (("val", sv), ("test", st)):
            print(f"{budget:6.1%} | {g[0]:5.2f} {g[1]:5.3f} {g[2]:5.2f} | {name:5} {s['human_lost']:6.2%} "
                  f"{s['machine_to_human']:6.2%} {s['unknown']:6.2%} {s['cause_acc']:6.2%} {s['mach_ms']:7.0f} {s['early_share']:6.1%}")
    json.dump(out, open(f"{a.early}/policy.json", "w"), indent=1)


if __name__ == "__main__":
    main()
