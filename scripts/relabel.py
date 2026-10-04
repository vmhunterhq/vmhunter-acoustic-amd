"""Fix known label errors in a cache dir's CSVs (audio .npy files are shared via symlink).

Rule 1: the old engine labelled the call-screening prompt ("Hi, if you record your
name and reason for calling...") as machine_speech via MARKERS_MACHINE:you. Its
transcripts are the same as rows labelled `screening`, so move them to `screening`.
"Reason for your call" alone is not used: ordinary voicemail greetings say it too.

Adds `orig_coarse` and `relabel` columns.
"""
import argparse, os
import pandas as pd

SCREENING_RX = r"you record your|record your name"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="cache/v2split")
    ap.add_argument("--dst", default="cache/v2split_relabel")
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    for split in ("train", "val", "test"):
        df = pd.read_csv(os.path.join(a.src, f"{split}.csv"), dtype=str, keep_default_na=False)
        df["orig_coarse"] = df.coarse
        df["relabel"] = ""
        hit = (df.coarse == "machine_speech") & df.transcript.str.lower().str.contains(SCREENING_RX)
        df.loc[hit, "coarse"] = "screening"
        df.loc[hit, "relabel"] = "screening_phrase"
        df.to_csv(os.path.join(a.dst, f"{split}.csv"), index=False)
        npy = os.path.join(a.dst, f"{split}.npy")
        if not os.path.exists(npy):
            os.symlink(os.path.join(a.src, f"{split}.npy"), npy)
        print(split, "machine_speech -> screening:", int(hit.sum()),
              "| tier B classes:", df[df.tier == "B"].coarse.value_counts().to_dict())


if __name__ == "__main__":
    main()
