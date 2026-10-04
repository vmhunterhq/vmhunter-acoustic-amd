# Dataset format

The audio and transcripts used to train these models are real phone calls and are not published. This file describes the format so you can build the same dataset from your own recordings.

## Clips

One WAV per answered call: 8 kHz, 16-bit, mono, starting at the moment of answer, at least 2.0 s (only the first 2.0 s is used). Ours were 518,693 unique clips.

## manifest.csv

One row per clip. Do not put phone numbers or caller IDs in it.

| Column | Meaning |
|---|---|
| `id` | unique row id |
| `source` | opaque id of the customer or campaign (`src01`, `src02`, ...) |
| `ts`, `day` | when the call was answered (unix time, `YYYYMMDD`) |
| `engine` | which version of your labelling detector produced the verdict (`v1`, `v2`) |
| `path` | path to the WAV |
| `md5` | hash of the audio bytes with the WAV header skipped, for duplicate removal |
| `transcript` | optional; used only by `relabel.py` for label cleanup |
| `amdreason` | optional; why the labelling detector decided as it did |
| `coarse` | the class: `human`, `machine_speech`, `screening`, `silence`, `tone` (or `noise`, not trained) |
| `tier` | label confidence: `A`, `B`, `C` or `D` |

## Classes

| `coarse` | What is in the audio |
|---|---|
| `human` | a live person: a short utterance, then a pause |
| `machine_speech` | a recording: voicemail greeting, carrier prompt, IVR |
| `screening` | a call-screening prompt (iPhone, Google, call blockers) |
| `silence` | no audio, or line hiss only |
| `tone` | a single tone: beep, ringback, SIT |

## Label tiers

Labels taken from an existing detector are its opinions, not ground truth. The tier says how far to trust each row. Only tier B (and A, if you have it) is used for training.

| Tier | Meaning | Use |
|---|---|---|
| A | verified by a person listening | test set, and training |
| B | a verdict rule that audits found reliable (for example voicemail wording with four or more words, a clear "Hello") | training |
| C | noisy: fallbacks, short transcripts, categories known to be wrong often | hold back |
| D | no decision was made | dropped |

## Rules that matter

1. **Remove duplicate audio** (`md5`). The same carrier recording appears thousands of times.
2. **Split by (source, day)**, not by row, so no customer-day is in both train and test. `scripts/resplit.py` does both.
3. **Read your labels.** `scripts/relabel.py` moved 37,101 clips from `machine_speech` to `screening` and took macro-F1 from 0.80 to 0.98 with no model change.
4. **Test on labels your labelling detector did not write.** Agent dispositions from the dialer are the best ground truth.
