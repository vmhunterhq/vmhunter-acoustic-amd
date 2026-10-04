"""Baseline acoustic AMD classifier: log-mel + small CNN, tier-B labels only.

Reads the cache from cache_audio.py. Picks the best epoch by val macro-F1,
then writes test metrics, a confusion matrix and per-clip predictions to --out.
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

SR = 8000
PREFIXES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]  # decision points (s) evaluated for early exit
CLASSES = ["human", "machine_speech", "screening", "silence", "tone"]
MACHINE = {"machine_speech", "screening"}  # recordings: the dialer must not send these to an agent


class LogMel(nn.Module):
    def __init__(self):
        super().__init__()
        # 32 ms window, 10 ms hop, 64 mels over the telephone band
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=8000, n_fft=256, win_length=256, hop_length=80, f_min=50, f_max=3800, n_mels=64)

    def forward(self, x):  # x: B x T float in [-1, 1]
        return torch.log(self.mel(x) + 1e-6).unsqueeze(1)  # B x 1 x 64 x frames


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class AMDNet(nn.Module):
    def __init__(self, n_classes, width=32):
        super().__init__()
        self.front = LogMel()
        self.norm = nn.BatchNorm2d(1)
        w = width
        self.body = nn.Sequential(
            block(1, w), nn.MaxPool2d(2),
            block(w, 2 * w), nn.MaxPool2d(2),
            block(2 * w, 4 * w), nn.MaxPool2d((2, 1)),  # keep time resolution late
            block(4 * w, 4 * w))
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(8 * w, n_classes))

    def forward(self, wav, specaug=False):
        x = self.norm(self.front(wav))
        if specaug:
            x = spec_augment(x)
        x = self.body(x).mean(2)  # pool frequency -> B x C x frames
        x = torch.cat([x.mean(2), x.amax(2)], 1)
        return self.head(x)


def ds_block(cin, cout, stride):
    """Depthwise-separable conv: ~8x cheaper than a full 3x3 conv."""
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False), nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
        nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class AMDLite(nn.Module):
    """~32k-parameter CNN for CPU serving at thousands of concurrent calls."""
    def __init__(self, n_classes, width=24):
        super().__init__()
        self.front = LogMel()
        self.norm = nn.BatchNorm2d(1)
        w = width
        self.body = nn.Sequential(
            nn.Conv2d(1, w, 3, 2, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
            ds_block(w, 2 * w, 2), ds_block(2 * w, 2 * w, 1),
            ds_block(2 * w, 4 * w, (2, 1)), ds_block(4 * w, 4 * w, 1), ds_block(4 * w, 4 * w, (2, 1)))
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(8 * w, n_classes))

    def forward(self, wav, specaug=False):
        x = self.norm(self.front(wav))
        if specaug:
            x = spec_augment(x)
        x = self.body(x).mean(2)
        x = torch.cat([x.mean(2), x.amax(2)], 1)
        return self.head(x)


ARCHS = {"cnn": AMDNet, "lite": AMDLite}


def build(arch, n_classes, width=None):
    return ARCHS[arch](n_classes) if width is None else ARCHS[arch](n_classes, width)


def spec_augment(x, f=8, t=20):
    B, _, Fq, T = x.shape
    f0 = torch.randint(0, Fq - f, (B, 1, 1, 1), device=x.device)
    fw = torch.randint(0, f + 1, (B, 1, 1, 1), device=x.device)
    t0 = torch.randint(0, T - t, (B, 1, 1, 1), device=x.device)
    tw = torch.randint(0, t + 1, (B, 1, 1, 1), device=x.device)
    fi = torch.arange(Fq, device=x.device).view(1, 1, -1, 1)
    ti = torch.arange(T, device=x.device).view(1, 1, 1, -1)
    mask = ((fi >= f0) & (fi < f0 + fw)) | ((ti >= t0) & (ti < t0 + tw))
    return x.masked_fill(mask, x.mean())


def load_split(cache, split, tier_b_only):
    df = pd.read_csv(os.path.join(cache, f"{split}.csv"), dtype=str, keep_default_na=False)
    wav = np.load(os.path.join(cache, f"{split}.npy"), mmap_mode="r")
    keep = df.coarse.isin(CLASSES).to_numpy().copy()
    if tier_b_only:
        keep &= (df.tier == "B").to_numpy()
    idx = np.flatnonzero(keep)
    df = df.iloc[idx].reset_index(drop=True)
    x = torch.from_numpy(np.ascontiguousarray(wav[idx]))
    y = torch.tensor(df.coarse.map(CLASSES.index).to_numpy())
    return df, x, y


@torch.no_grad()
def predict(model, x, dev, bs=2048, keep_s=None):
    """Class probabilities; with keep_s, only the first keep_s seconds are heard (rest zeroed, like a live buffer)."""
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs].to(dev, non_blocking=True).float() / 32768
        if keep_s is not None:
            xb[:, int(keep_s * SR):] = 0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(xb).float().softmax(1).cpu())
    return torch.cat(out)


def metrics(y, p):
    y, pred = y.numpy(), p.argmax(1).numpy()
    n = len(CLASSES)
    cm = np.zeros((n, n), int)
    np.add.at(cm, (y, pred), 1)
    per = {}
    for i, c in enumerate(CLASSES):
        tp, sup, npred = cm[i, i], cm[i].sum(), cm[:, i].sum()
        prec = tp / npred if npred else float("nan")
        rec = tp / sup if sup else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if sup and npred and prec + rec else 0.0
        per[c] = {"precision": prec, "recall": rec, "f1": f1 if sup else float("nan"), "support": int(sup)}
    present = [c for c in CLASSES if per[c]["support"]]
    hum = CLASSES.index("human")
    mach = [CLASSES.index(c) for c in MACHINE]
    return {
        "macro_f1": float(np.mean([per[c]["f1"] for c in present])),
        "per_class": per,
        # the two business errors
        "human_lost_rate": float(1 - cm[hum, hum] / cm[hum].sum()),  # live person not passed as human
        "human_to_machine_rate": float(cm[hum, mach].sum() / cm[hum].sum()),
        "machine_to_human_rate": float(cm[mach, hum].sum() / cm[mach].sum()),  # recording sent to an agent
        "human_precision": per["human"]["precision"],
        "confusion": cm.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/v2split")
    ap.add_argument("--out", default="runs/baseline")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--arch", choices=list(ARCHS), default="cnn")
    ap.add_argument("--width", type=int, default=None, help="default: 32 for cnn, 24 for lite")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix-prob", type=float, default=0.0,
                    help="share of training clips cut to a random 0.5-2.0 s prefix (zero-padded), for early decisions")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = "cuda"

    tr_df, xtr, ytr = load_split(a.cache, "train", True)
    va_df, xva, yva = load_split(a.cache, "val", True)
    xtr, ytr_dev = xtr.to(dev), ytr.to(dev)  # ~7 GB int16, fits on a 24 GB GPU
    print("train", np.bincount(ytr, minlength=len(CLASSES)), "val", np.bincount(yva, minlength=len(CLASSES)), flush=True)

    # sample classes in proportion to sqrt(count): softens the 70 % machine_speech skew
    cnt = torch.bincount(ytr, minlength=len(CLASSES)).float()
    w = (cnt.clamp(min=1) ** -0.5)[ytr]
    sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=len(ytr), replacement=True)

    model = build(a.arch, len(CLASSES), a.width).to(dev)
    print("params", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    steps = a.epochs * (len(ytr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)

    best, history = -1, []
    for ep in range(a.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        order = torch.tensor(list(sampler), device=dev)
        for i in range(len(ytr) // a.bs):
            bi = order[i * a.bs:(i + 1) * a.bs]
            xb = xtr[bi].to(dev, non_blocking=True).float() / 32768
            xb = xb * (10 ** (torch.empty(len(xb), 1, device=dev).uniform_(-0.5, 0.5)))  # +-10 dB gain
            if a.prefix_prob > 0:
                n = len(xb)
                keep = torch.randint(int(0.5 * SR), xb.shape[1] + 1, (n, 1), device=dev)
                cut = torch.rand(n, 1, device=dev) < a.prefix_prob
                pos = torch.arange(xb.shape[1], device=dev).view(1, -1)
                xb = xb.masked_fill(cut & (pos >= keep), 0)
            yb = ytr_dev[bi]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(xb, specaug=True), yb, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        m = metrics(yva, predict(model, xva, dev))
        score = m["macro_f1"]
        if a.prefix_prob > 0:  # select for both early (1.0 s) and full-window quality
            m1 = metrics(yva, predict(model, xva, dev, keep_s=1.0))
            score = (m["macro_f1"] + m1["macro_f1"]) / 2
        row = {"epoch": ep + 1, "loss": tot / (len(ytr) // a.bs), "val_macro_f1": m["macro_f1"], "score": score,
               "val_human_lost": m["human_lost_rate"], "val_machine_to_human": m["machine_to_human_rate"],
               "sec": round(time.time() - t0, 1)}
        history.append(row)
        print(json.dumps(row), flush=True)
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "classes": CLASSES, "args": vars(a), "epoch": ep + 1},
                       os.path.join(a.out, "best.pt"))

    ck = torch.load(os.path.join(a.out, "best.pt"))
    model.load_state_dict(ck["model"])
    report = {"best_epoch": ck["epoch"], "history": history}
    te_df, xte, yte = load_split(a.cache, "test", True)
    for name, df, x, y in [("val", va_df, xva, yva), ("test", te_df, xte, yte)]:
        p = predict(model, x, dev)
        report[name] = metrics(y, p)
        for eng in ("v1", "v2"):
            sel = torch.from_numpy((df.engine == eng).to_numpy())
            if sel.any():
                report[f"{name}_{eng}"] = metrics(y[sel], p[sel])
        if name == "test":
            out = df[["id", "source", "day", "engine", "tier", "coarse", "amdreason", "transcript", "path"]].copy()
            out["pred"] = [CLASSES[i] for i in p.argmax(1)]
            out["conf"] = p.max(1).values.numpy().round(4)
            for i, c in enumerate(CLASSES):
                out[f"p_{c}"] = p[:, i].numpy().round(4)
            out.to_csv(os.path.join(a.out, "test_predictions.csv"), index=False)
        # probabilities at every decision point, for tuning early-exit thresholds (early_exit.py)
        probs = np.stack([predict(model, x, dev, keep_s=t).numpy() for t in PREFIXES])  # P x N x C
        np.save(os.path.join(a.out, f"probs_{name}.npy"), probs.astype(np.float32))
        np.save(os.path.join(a.out, f"labels_{name}.npy"), y.numpy())
        df[["id", "engine", "coarse", "transcript"]].to_csv(os.path.join(a.out, f"rows_{name}.csv"), index=False)
        for t, pt in zip(PREFIXES, probs):
            report[f"{name}@{t}s"] = metrics(y, torch.from_numpy(pt))
    json.dump(report, open(os.path.join(a.out, "report.json"), "w"), indent=1, default=float)

    print("\n== test by decision time (argmax, no thresholds)")
    for t in PREFIXES:
        r = report[f"test@{t}s"]
        print(f"  {t:4.2f}s macro-F1 {r['macro_f1']:.4f} | human lost {r['human_lost_rate']:.4f} "
              f"| machine->human {r['machine_to_human_rate']:.4f}")

    for k in ("val", "test", "test_v1", "test_v2"):
        if k not in report:
            continue
        r = report[k]
        print(f"\n== {k}: macro-F1 {r['macro_f1']:.4f} | human lost {r['human_lost_rate']:.4f} "
              f"(to machine {r['human_to_machine_rate']:.4f}) | machine->human {r['machine_to_human_rate']:.4f}")
        for c, v in r["per_class"].items():
            print(f"  {c:15s} P {v['precision']:.4f} R {v['recall']:.4f} F1 {v['f1']:.4f} n {v['support']}")
        print("  confusion (rows true, cols pred):", CLASSES)
        for c, row in zip(CLASSES, r["confusion"]):
            print(f"  {c:15s}", row)


if __name__ == "__main__":
    main()
