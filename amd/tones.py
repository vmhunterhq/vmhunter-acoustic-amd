"""Streaming call-progress tone detector for 8 kHz telephone audio.

Pure DSP, no training data needed: these tones are fixed frequency patterns.
Feed audio with `push()` as it arrives; it returns a verdict the first time a
pattern is confirmed, else None.

  BUSY: busy (480+620 Hz, 0.5 s on/off) and reorder (same pair, 0.25 s on/off)
  TONE: SIT (three ascending tones ~914/1371/1777 Hz), ringback (440+480),
        dial tone (350+440), fax CNG (1100 Hz) / CED (2100 Hz), and any other
        steady single tone such as a voicemail beep.
"""
from dataclasses import dataclass, field
import numpy as np

SR = 8000
HOP = 160          # 20 ms
WIN = 400          # 50 ms analysis window (~20 Hz resolution)
NFFT = 512
FLOOR_DB = -45.0   # frames quieter than this (dBFS) are treated as silence

# dual-tone pairs (Hz) -> name
DUALS = {"busy": (480, 620), "ringback": (440, 480), "dial": (350, 440)}
SIT_BANDS = [(880, 1010), (1340, 1460), (1740, 1810)]  # low / mid / high segment of a SIT
FAX_CNG, FAX_CED = 1100, 2100

_freqs = np.fft.rfftfreq(NFFT, 1 / SR)
_band = (_freqs >= 250) & (_freqs <= 3500)
_window = np.hanning(WIN)


def _bin_power(P, f, width=1):
    k = int(round(f * NFFT / SR))
    return P[max(k - width, 0):k + width + 1].sum()


def classify_frame(frame):
    """Label one 50 ms frame: 'silence', 'other', 'dual:<name>' or 'single:<Hz>'."""
    x = frame.astype(np.float64) / 32768
    rms_db = 10 * np.log10(np.mean(x ** 2) + 1e-12)
    if rms_db < FLOOR_DB:
        return "silence"
    P = np.abs(np.fft.rfft(x * _window, NFFT)) ** 2
    E = P[_band].sum() + 1e-20
    for name, (f1, f2) in DUALS.items():
        a, b = _bin_power(P, f1), _bin_power(P, f2)
        if a / E > 0.2 and b / E > 0.2 and (a + b) / E > 0.75:
            return f"dual:{name}"
    k = int(np.argmax(np.where(_band, P, 0)))
    if P[max(k - 2, 0):k + 3].sum() / E > 0.8:  # one dominant line
        # refine the peak (parabolic interpolation on log power): ~1 Hz accuracy for a pure tone
        a, b, c = np.log(P[k - 1:k + 2] + 1e-20)
        f = (k + 0.5 * (a - c) / (a - 2 * b + c + 1e-12)) * SR / NFFT
        # a voice has harmonics; a generated tone has none
        if 2 * f < 3500 and _bin_power(P, 2 * f) / E > 0.01:
            return "other"
        return f"single:{f:.1f}"
    return "other"


@dataclass
class Segment:
    label: str
    start: int   # frame index
    end: int     # exclusive
    freqs: list = field(default_factory=list)  # per-frame frequency, single tones only


@dataclass
class ToneDetector:
    segments: list = field(default_factory=list)
    _buf: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int16))
    _frames: int = 0
    verdict: tuple = None   # (cause, detail, decided_at_ms)

    def push(self, pcm):
        """pcm: int16 samples (any length). Returns the verdict once confirmed, else None."""
        if self.verdict:
            return self.verdict
        self._buf = np.concatenate([self._buf, np.asarray(pcm, np.int16)])
        while len(self._buf) >= WIN:
            lab = classify_frame(self._buf[:WIN])
            self._buf = self._buf[HOP:]
            self._add(lab)
            self._frames += 1
            v = self._check()
            if v:
                self.verdict = (*v, self._frames * HOP * 1000 // SR + (WIN - HOP) * 1000 // SR)
                return self.verdict
        return None

    def _add(self, lab):
        key = lab
        if lab.startswith("single:"):
            f = float(lab.split(":")[1])
            # merge frames of the same steady tone: <= 6 Hz step, <= 15 Hz from the segment start
            last = self.segments[-1] if self.segments else None
            if last and last.label.startswith("single:") and last.end == self._frames \
                    and abs(last.freqs[-1] - f) <= 6 and abs(last.freqs[0] - f) <= 15:
                last.end += 1
                last.freqs.append(f)
                return
            self.segments.append(Segment(f"single:{round(f)}", self._frames, self._frames + 1, [f]))
            return
        if self.segments and self.segments[-1].label == key and self.segments[-1].end == self._frames:
            self.segments[-1].end += 1
        else:
            self.segments.append(Segment(key, self._frames, self._frames + 1))

    def _dur(self, seg):
        return (seg.end - seg.start) * HOP / SR  # seconds

    def _check(self):
        segs = self.segments
        on = lambda name: [s for s in segs if s.label == f"dual:{name}"]
        busy = on("busy")
        if sum(self._dur(s) for s in busy) >= 0.6 or (len(busy) >= 2 and all(self._dur(s) >= 0.15 for s in busy[:2])):
            return ("BUSY", "busy" if max(self._dur(s) for s in busy) > 0.35 else "reorder")
        if sum(self._dur(s) for s in on("ringback")) >= 0.8:
            return ("TONE", "ringback")
        if sum(self._dur(s) for s in on("dial")) >= 0.8:
            return ("TONE", "dial")

        singles = [(int(s.label.split(":")[1]), s) for s in segs if s.label.startswith("single:")]
        # SIT: at least two of its three tones, in ascending order, each 0.15-0.5 s
        sit_seq = []
        for f, s in singles:
            for i, (lo, hi) in enumerate(SIT_BANDS):
                if lo <= f <= hi and 0.15 <= self._dur(s) <= 0.5:
                    sit_seq.append(i)
        if any(b == a + 1 for a, b in zip(sit_seq, sit_seq[1:])):
            return ("TONE", "sit")
        for f, s in singles:
            d = self._dur(s)
            if abs(f - FAX_CNG) <= 30 and d >= 0.4:
                return ("TONE", "fax")
            if abs(f - FAX_CED) <= 30 and d >= 0.5:
                return ("TONE", "fax")
            # a steady pure tone of 0.4 s or more. Shorter pips (e.g. the ~0.18 s 523 Hz
            # tone heard before many live "Hello?"s) are played at connect and say nothing.
            if 300 <= f <= 2600 and d >= 0.4:
                if not any(lo <= f <= hi for lo, hi in SIT_BANDS) or d > 0.5:
                    return ("TONE", f"beep:{f}")
        return None


def detect(pcm):
    """Run over a whole clip; returns (cause, detail, decided_at_ms) or None."""
    det = ToneDetector()
    for i in range(0, len(pcm), HOP):
        v = det.push(pcm[i:i + HOP])
        if v:
            return v
    return None
