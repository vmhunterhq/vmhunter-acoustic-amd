"""Load every split's WAVs into one int16 array per split (fixed 2.0 s window).

Clips longer than 2.0 s keep their first 2.0 s (the dialer wants an early
decision); shorter clips are zero-padded at the end and `n_samples` records the
real length. Writes <out>/<split>.npy (N x 16000 int16) and <split>.csv.
"""
import argparse, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import soundfile as sf

SR, WIN = 8000, 16000


def load(path):
    x, sr = sf.read(path, dtype="int16", frames=WIN)
    assert sr == SR
    out = np.zeros(WIN, np.int16)
    out[: len(x)] = x
    return out, len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", default="data/v2split")
    ap.add_argument("--out", default="cache/v2split")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for split in ("train", "val", "test"):
        df = pd.read_csv(os.path.join(a.split_dir, f"{split}.csv"), dtype=str, keep_default_na=False)
        with ProcessPoolExecutor(32) as ex:
            res = list(ex.map(load, df.path, chunksize=1000))
        np.save(os.path.join(a.out, f"{split}.npy"), np.stack([r[0] for r in res]))
        df["n_samples"] = [r[1] for r in res]
        df.to_csv(os.path.join(a.out, f"{split}.csv"), index=False)
        print(split, len(df), flush=True)


if __name__ == "__main__":
    main()
