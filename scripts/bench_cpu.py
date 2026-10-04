"""CPU throughput of the CNN (2 s windows) and the tone detector, for capacity planning."""
import os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from train import AMDNet, CLASSES  # noqa: E402
from amd.tones import ToneDetector, HOP  # noqa: E402

torch.set_grad_enabled(False)
m = AMDNet(len(CLASSES)).eval()
x = torch.randn(256, 16000) * 0.1
for threads in (1, 4, 8):
    torch.set_num_threads(threads)
    for bs in (1, 64, 256):
        xb = x[:bs]
        m(xb)
        n, t0 = 0, time.time()
        while time.time() - t0 < 3:
            m(xb); n += bs
        print(f"cnn threads={threads} batch={bs}: {n / (time.time() - t0):8.0f} windows/s")

# tone detector: one call's 2 s of audio, 20 ms pushes (pure python/numpy per call)
pcm = (np.random.randn(16000) * 1000).astype(np.int16)
n, t0 = 0, time.time()
while time.time() - t0 < 3:
    d = ToneDetector()
    for i in range(0, 16000, HOP):
        d.push(pcm[i:i + HOP])
    n += 1
print(f"tone detector, 1 core: {n / (time.time() - t0):.0f} calls (2 s each) per second")
