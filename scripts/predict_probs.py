"""Save a trained run's probabilities at every decision point for val and test.

Same outputs as the end of train.py (probs_{split}.npy, labels_{split}.npy), for
runs trained before that was added. Row order matches train.load_split.
"""
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from train import CLASSES, PREFIXES, build, load_split, predict  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cache", default="cache/v2split_relabel")
    a = ap.parse_args()
    ck = torch.load(os.path.join(a.run, "best.pt"))
    model = build(ck["args"].get("arch", "cnn"), len(CLASSES), ck["args"]["width"]).cuda()
    model.load_state_dict(ck["model"])
    for split in ("val", "test"):
        _, x, y = load_split(a.cache, split, True)
        probs = np.stack([predict(model, x, "cuda", keep_s=t).numpy() for t in PREFIXES])
        np.save(os.path.join(a.run, f"probs_{split}.npy"), probs.astype(np.float32))
        np.save(os.path.join(a.run, f"labels_{split}.npy"), y.numpy())
        print(split, probs.shape)


if __name__ == "__main__":
    main()
