"""Classify WAV files with the trained model: no server, no GPU.

    pip install onnxruntime numpy
    python scripts/classify_wav.py call1.wav call2.wav
    python scripts/classify_wav.py --model models/lite/amd.onnx --human-threshold 0.5 *.wav

Each file must be 8 kHz, 16-bit, mono, starting at the moment the call is answered.
Only the first 2.0 s is used (shorter files are zero-padded), as the server does.
Convert other formats first:  sox in.wav -r 8000 -c 1 -b 16 out.wav
"""
import argparse, json, os, sys, wave
from types import SimpleNamespace
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from amd.server import WINDOW, decide  # noqa: E402
from amd.tones import detect  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", nargs="+")
    ap.add_argument("--model", default="models/full/amd.onnx")
    ap.add_argument("--human-threshold", type=float, default=0.4)
    ap.add_argument("--min-confidence", type=float, default=0.5)
    a = ap.parse_args()
    classes = json.load(open(os.path.join(os.path.dirname(a.model), "amd.json")))["classes"]
    sess = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
    opts = SimpleNamespace(human_threshold=a.human_threshold, min_confidence=a.min_confidence)
    for path in a.wav:
        with wave.open(path) as w:
            if (w.getframerate(), w.getsampwidth(), w.getnchannels()) != (8000, 2, 1):
                print(f"{path}: skipped, need 8 kHz 16-bit mono "
                      f"(got {w.getframerate()} Hz, {8 * w.getsampwidth()}-bit, {w.getnchannels()} ch)")
                continue
            got = np.frombuffer(w.readframes(WINDOW), np.int16)
        pcm = np.zeros(WINDOW, np.int16)
        pcm[: len(got)] = got
        probs = sess.run(None, {"pcm": pcm[None]})[0][0]
        cause, conf, tone = decide(probs, classes, detect(got), opts)
        status = "HUMAN" if cause == "HUMAN" else "MACHINE"
        detail = " ".join(f"{c}={p:.2f}" for c, p in zip(classes, probs))
        print(f"{path}: {status} / {cause} (confidence {conf:.2f}{', tone ' + tone if tone else ''})  [{detail}]")


if __name__ == "__main__":
    main()
