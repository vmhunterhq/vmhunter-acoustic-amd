"""Re-split the dataset so every class is measurable in val and test.

- Drops tier D and exact-duplicate audio (same md5; keeps the earliest clip).
- Groups by (source, day): no customer-day appears in two splits.
- Random search over group assignments. v2 groups are favoured for test
  (better labels). Score: how far each class's tier-B share in val and test
  is from the target, with a penalty when a class has too few test clips.

Writes <out>/{train,val,test}.csv (all tiers, with `split`) and resplit_summary.txt.
"""
import argparse, os
import numpy as np
import pandas as pd

CLASSES = ["human", "machine_speech", "screening", "silence", "tone", "noise"]
B_CLASSES = ["human", "machine_speech", "screening", "silence", "tone"]  # classes with tier-B rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out", default="data/v2split")
    ap.add_argument("--val", type=float, default=0.10)
    ap.add_argument("--test", type=float, default=0.10)
    ap.add_argument("--min-test", type=int, default=60, help="min tier-B test clips per class")
    ap.add_argument("--tries", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    m = pd.read_csv(a.manifest, dtype=str, keep_default_na=False)
    n0 = len(m)
    m = m[m.tier != "D"].sort_values("ts", kind="stable").drop_duplicates("md5")
    m["group"] = m.source + "|" + m.day

    b = m[m.tier == "B"]
    counts = pd.crosstab(b.group, b.coarse).reindex(columns=B_CLASSES, fill_value=0)
    groups = sorted(m.group.unique())
    counts = counts.reindex(groups, fill_value=0)
    C = counts.to_numpy(float)  # groups x classes
    total = C.sum(0)
    is_v2 = m.groupby("group").engine.agg(lambda s: (s == "v2").mean() > 0.5).reindex(groups).to_numpy()

    # assignment probabilities: v2 groups lean towards test
    p_v1 = np.array([1 - a.val - a.test, a.val, a.test])
    p_v2 = np.array([0.35, 0.15, 0.50])
    target = np.array([a.val, a.test])

    best, best_score = None, np.inf
    for _ in range(a.tries):
        u = rng.random(len(groups))
        cum = np.where(is_v2[:, None], np.cumsum(p_v2), np.cumsum(p_v1))
        assign = (u[:, None] > cum).sum(1)  # 0 train, 1 val, 2 test
        frac = np.stack([C[assign == k].sum(0) / total for k in (1, 2)])  # 2 x classes
        score = np.abs(frac - target[:, None]).sum() / target.mean()
        test_n = C[assign == 2].sum(0)
        score += 5 * np.clip((a.min_test - test_n) / a.min_test, 0, None).sum()
        score += 5 * np.clip((a.min_test - C[assign == 1].sum(0)) / a.min_test, 0, None).sum()
        if score < best_score:
            best, best_score = assign, score

    split_of = dict(zip(groups, np.array(["train", "val", "test"])[best]))
    m["split"] = m.group.map(split_of)
    m = m.drop(columns="group")
    for s in ("train", "val", "test"):
        m[m.split == s].to_csv(os.path.join(a.out, f"{s}.csv"), index=False)

    lines = [f"manifest rows: {n0}; after dropping tier D and duplicates: {len(m)}; score {best_score:.3f}", ""]
    tab = pd.crosstab([m.coarse, m.tier], m.split).reindex(columns=["train", "val", "test"], fill_value=0)
    lines.append(tab.to_string())
    lines.append("\n[engine share by split]")
    lines.append(pd.crosstab(m.split, m.engine, normalize="index").round(3).to_string())
    lines.append("\n[groups by split]")
    lines.append(pd.Series(split_of).value_counts().to_string())
    lines.append("\n[test groups]")
    lines.append(m[m.split == "test"].groupby(["source", "day"]).size().to_string())
    txt = "\n".join(lines)
    open(os.path.join(a.out, "resplit_summary.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
