"""Evaluate amd/tones.py.

1. Real clips: run over every val+test clip (all tiers) from the cache; report
   detection rate per (class, engine reason). Detections on `human` clips are
   the costly false positives.
2. Synthetic: busy, reorder, SIT, ringback, fax, beeps with random level, noise
   and mu-law coding, starting at a random offset; report detection rate and time.
"""
import argparse, os, sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from amd.tones import detect, SR  # noqa: E402

WIN = 16000


def _run(chunk):
    return [detect(x) for x in chunk]


def run_all(x, workers=32):
    chunks = np.array_split(np.arange(len(x)), workers * 8)
    with ProcessPoolExecutor(workers) as ex:
        res = ex.map(_run, [np.asarray(x[c]) for c in chunks])
    return [v for r in res for v in r]


def mulaw(x):  # round trip through G.711 mu-law, like the phone network
    mu = 255.0
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    y = np.round(y * 127) / 127
    return np.sign(y) * ((1 + mu) ** np.abs(y) - 1) / mu


def tone(freqs, dur, rng):
    t = np.arange(int(dur * SR)) / SR
    return sum(np.sin(2 * np.pi * f * rng.uniform(0.99, 1.01) * t + rng.uniform(0, 6.28)) for f in freqs) / len(freqs)


def cadence(freqs, on, off, total, rng):
    out = []
    while sum(map(len, out)) < total * SR:
        out += [tone(freqs, on, rng), np.zeros(int(off * SR))]
    return np.concatenate(out)


def synth(kind, rng):
    if kind == "busy":
        s = cadence((480, 620), 0.5, 0.5, 3, rng)
    elif kind == "reorder":
        s = cadence((480, 620), 0.25, 0.25, 3, rng)
    elif kind == "ringback":
        s = cadence((440, 480), 2.0, 4.0, 3, rng)
    elif kind == "sit":
        lo, mid = rng.choice([913.8, 985.2]), rng.choice([1370.6, 1428.5])
        d = [rng.choice([0.276, 0.380]) for _ in range(3)]
        s = np.concatenate([tone((lo,), d[0], rng), tone((mid,), d[1], rng), tone((1776.7,), d[2], rng), np.zeros(SR)])
    elif kind == "fax_cng":
        s = cadence((1100,), 0.5, 3.0, 3, rng)
    elif kind == "beep":
        s = np.concatenate([tone((rng.uniform(400, 2400),), rng.uniform(0.4, 1.2), rng), np.zeros(SR)])
    elif kind == "pip":  # short connect tone before a live answer: must NOT be detected
        s = np.concatenate([tone((523.25,), rng.uniform(0.12, 0.25), rng), np.zeros(SR)])
    else:
        raise ValueError(kind)
    level = 10 ** (rng.uniform(-30, -8) / 20)
    s = s * level
    lead = np.zeros(int(rng.uniform(0, 0.8) * SR))
    x = np.concatenate([lead, s])[:WIN]
    x = np.pad(x, (0, WIN - len(x)))
    snr = rng.uniform(10, 35)
    noise = rng.standard_normal(WIN) * level / np.sqrt(2) * 10 ** (-snr / 20)
    return (np.clip(mulaw(np.clip(x + noise, -1, 1)), -1, 1) * 32767).astype(np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/v2split_relabel")
    ap.add_argument("--out", default="runs/tones")
    ap.add_argument("--n-synth", type=int, default=500)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    print("== synthetic (expected cause in brackets)")
    rng = np.random.default_rng(0)
    expect = {"busy": "BUSY", "reorder": "BUSY", "ringback": "TONE", "sit": "TONE", "fax_cng": "TONE", "beep": "TONE", "pip": None}
    for kind, cause in expect.items():
        x = np.stack([synth(kind, rng) for _ in range(a.n_synth)])
        res = run_all(x)
        hit = [r for r in res if r and (r[0] == cause or cause is None)]
        details = Counter(r[1].split(":")[0] for r in res if r)
        t = np.median([r[2] for r in hit]) if hit else float("nan")
        print(f"  {kind:9s} [{cause}] detected {len(hit) / len(res):6.1%}  median at {t:5.0f} ms  details {dict(details)}")

    print("\n== real clips (val + test, all tiers)")
    rows = []
    for split in ("val", "test"):
        df = pd.read_csv(os.path.join(a.cache, f"{split}.csv"), dtype=str, keep_default_na=False)
        x = np.load(os.path.join(a.cache, f"{split}.npy"), mmap_mode="r")
        res = run_all(x)
        df["tone_cause"] = [r[0] if r else "" for r in res]
        df["tone_detail"] = [r[1] if r else "" for r in res]
        df["tone_ms"] = [r[2] if r else "" for r in res]
        df["split"] = split
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    df["reason"] = df.amdreason.str.split(":").str[0].str.replace(r"_\d+W$", "", regex=True)
    df[df.tone_cause != ""].to_csv(os.path.join(a.out, "real_detections.csv"), index=False)
    g = df.groupby(["coarse", "tier", "reason"]).agg(n=("id", "size"), detected=("tone_cause", lambda s: (s != "").mean()))
    g = g[g.n >= 20].sort_values("detected", ascending=False)
    print(g.to_string(float_format=lambda v: f"{v:.3%}"))
    print("\n[detail of detections by class]")
    print(pd.crosstab(df.coarse, df.tone_detail.str.split(":").str[0]).to_string())


if __name__ == "__main__":
    main()
