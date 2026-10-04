"""Export a trained run to ONNX: int16 PCM (N x 16000) in, class probabilities out.

torchaudio's STFT does not export cleanly, so the log-mel front end is rebuilt from
plain ops (reflect pad, frame, window, DFT as matmul, mel matmul, log) and checked
against the training model before writing. Writes <out>/amd.onnx and amd.json.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from train import CLASSES, build  # noqa: E402


class ExportFront(nn.Module):
    """Same output as train.LogMel (torchaudio MelSpectrogram, center=True, reflect pad, power 2)."""
    def __init__(self, logmel):
        super().__init__()
        spec = logmel.mel.spectrogram
        self.n_fft, self.hop = spec.n_fft, spec.hop_length
        n = torch.arange(self.n_fft, dtype=torch.float64)
        k = torch.arange(self.n_fft // 2 + 1, dtype=torch.float64)
        ang = 2 * torch.pi * n[:, None] * k[None, :] / self.n_fft
        win = spec.window.double()[:, None]
        # windowed DFT basis as conv1d kernels: out channels = frequency bins
        self.register_buffer("cos", (win * torch.cos(ang)).T.unsqueeze(1).float())  # bins x 1 x n_fft
        self.register_buffer("sin", (win * torch.sin(ang)).T.unsqueeze(1).float())
        self.register_buffer("fb", logmel.mel.mel_scale.fb.float())  # (n_fft/2+1) x n_mels

    def forward(self, x):  # x: B x T float
        pad = self.n_fft // 2
        x = nn.functional.pad(x.unsqueeze(1), (pad, pad), mode="reflect")  # B x 1 x T'
        re = nn.functional.conv1d(x, self.cos, stride=self.hop)  # B x bins x F
        im = nn.functional.conv1d(x, self.sin, stride=self.hop)
        power = (re ** 2 + im ** 2).transpose(1, 2)  # B x F x bins
        mel = power @ self.fb  # B x F x mels
        return torch.log(mel + 1e-6).transpose(1, 2).unsqueeze(1)  # B x 1 x mels x F


class Exported(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.front = ExportFront(model.front)
        self.model = model

    def forward(self, pcm):  # int16 B x 16000
        m = self.model
        x = m.norm(self.front(pcm.float() / 32768))
        x = m.body(x).mean(2)
        x = torch.cat([x.mean(2), x.amax(2)], 1)
        return m.head(x).softmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None, help="default: <run>/export")
    ap.add_argument("--cache", default="cache/v2split_relabel")
    a = ap.parse_args()
    out = a.out or os.path.join(a.run, "export")
    os.makedirs(out, exist_ok=True)

    ck = torch.load(os.path.join(a.run, "best.pt"), map_location="cpu")
    model = build(ck["args"].get("arch", "cnn"), len(CLASSES), ck["args"]["width"])
    model.load_state_dict(ck["model"])
    model.eval()
    exp = Exported(model).eval()

    x = torch.from_numpy(np.ascontiguousarray(np.load(os.path.join(a.cache, "test.npy"), mmap_mode="r")[:512]))
    with torch.no_grad():
        ref = model(x.float() / 32768).softmax(1)
        got = exp(x)
    print("torch export front vs training model: max |dp| =", (ref - got).abs().max().item())

    path = os.path.join(out, "amd.onnx")
    torch.onnx.export(exp, (x[:4],), path, input_names=["pcm"], output_names=["probs"],
                      dynamic_axes={"pcm": {0: "batch"}, "probs": {0: "batch"}}, opset_version=17, dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    onx = sess.run(None, {"pcm": x.numpy()})[0]
    diff = np.abs(onx - ref.numpy()).max()
    agree = (onx.argmax(1) == ref.numpy().argmax(1)).mean()
    print(f"onnxruntime vs training model: max |dp| = {diff:.2e}, argmax agreement {agree:.4f}")
    assert diff < 1e-3, "export mismatch"
    meta = {"classes": CLASSES, "sample_rate": 8000, "window_samples": 16000, "input": "int16 pcm, batch x 16000",
            "run": a.run, "epoch": ck["epoch"], "arch": ck["args"].get("arch", "cnn")}
    json.dump(meta, open(os.path.join(out, "amd.json"), "w"), indent=1)
    print("wrote", path, os.path.getsize(path), "bytes")


if __name__ == "__main__":
    main()
