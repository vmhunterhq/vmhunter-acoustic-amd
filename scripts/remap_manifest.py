"""Point manifest/split paths at the extracted audio and check every file decodes.

Writes <out>/{manifest,train,val,test}.csv with `path` rewritten, the duplicated
`split` column removed, and a `audio_ok` column; prints a summary of problems.
"""
import argparse, os
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import soundfile as sf

PREFIX = "/var/recordings/"  # path prefix of the recordings as written in manifest.csv


def check(path):
    try:
        info = sf.info(path)
        return "ok" if (info.samplerate == 8000 and info.channels == 1) else f"fmt:{info.samplerate}/{info.channels}"
    except Exception as e:
        return "missing" if not os.path.exists(path) else f"bad:{type(e).__name__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--audio", default="audio")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    man = pd.read_csv(os.path.join(a.dataset, "manifest.csv"), dtype=str, keep_default_na=False)
    assert man.path.str.startswith(PREFIX).all(), "unexpected path prefix"
    man["path"] = a.audio + "/" + man.path.str[len(PREFIX):]
    with ProcessPoolExecutor(32) as ex:
        man["audio_ok"] = list(ex.map(check, man.path, chunksize=2000))
    man.to_csv(os.path.join(a.out, "manifest.csv"), index=False)
    print("manifest audio check:\n", man.audio_ok.value_counts().to_string())

    status = dict(zip(man.id, man.audio_ok))
    for split in ("train", "val", "test"):
        df = pd.read_csv(os.path.join(a.dataset, f"{split}.csv"), dtype=str, keep_default_na=False)
        df = df.loc[:, ~df.columns.duplicated()]
        df["path"] = a.audio + "/" + df.path.str[len(PREFIX):]
        df["audio_ok"] = df.id.map(status)
        df.to_csv(os.path.join(a.out, f"{split}.csv"), index=False)
        print(f"{split}: {len(df)} rows, not ok: {(df.audio_ok != 'ok').sum()}")


if __name__ == "__main__":
    main()
