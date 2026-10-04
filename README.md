# VM Hunter Acoustic AMD

An acoustic answering machine detection model for outbound dialers, with trained weights, training code, a WebSocket server and an Asterisk client. Built by [VM Hunter](https://vmhunter.com) as a research project and published with the article *How to Build and Train an Acoustic Answering Machine Detection Model*.

It decides from the first **2.0 s** of answered-call audio, using a CNN on log-mel spectrograms plus a DSP call-progress tone detector. No speech-to-text.

```
Asterisk 13+ ──EAGI fd 3──▶ amd-eagi (C, per call) ──WebSocket, slin16 8 kHz──▶ amd.server (ONNX, batched)
     ▲                                                                                │
     └──────── SET VARIABLE AMDSTATUS / AMDCAUSE ◀──────── JSON verdict at 2.0 s ◀────┘
```

| `AMDSTATUS` | `AMDCAUSE` | Meaning |
|---|---|---|
| `HUMAN` | `HUMAN` | live person |
| `MACHINE` | `MACHINE` | voicemail greeting, carrier prompt, IVR |
| `MACHINE` | `SCREENING` | iOS / Google call screening, call blockers |
| `MACHINE` | `SILENCE` | dead air |
| `MACHINE` | `TONE` | SIT, fax, ringback, beep, static |
| `MACHINE` | `BUSY` | busy / reorder |
| `MACHINE` | `UNKNOWN` | not confident, or any error / timeout |

## Read this first: what it can and cannot do

- **This is not the VM Hunter production engine.** The VM Hunter service is speech-based (it reads the words as well as the audio). This repository is the acoustic model we built to measure how far sound alone goes.
- **It is good at the easy majority**: long voicemail greetings, carrier prompts, call-screening prompts, silence and tones.
- **It is weak on short answers.** On 4,000 agent-labelled calls where the first word was a lone "Hello", it passed 78.7 % of answering machines to an agent (aligned subset). A person and a voicemail greeting that both start with "Hello" sound the same for two seconds. Full results: [Acoustic AMD vs speech-based AMD on 4,000 agent-labelled calls](https://vmhunter.com/blog/acoustic-amd-vs-speech-amd-4000-call-test).
- Trained on English-language US calls at 8 kHz. Test it on your own traffic before using it on live calls.

## Try it in two minutes

```bash
git clone https://github.com/vmhunterhq/vmhunter-acoustic-amd.git
cd vmhunter-acoustic-amd
pip install onnxruntime numpy
python scripts/classify_wav.py your_call.wav
```

Output:

```
your_call.wav: MACHINE / SCREENING (confidence 0.99)  [human=0.00 machine_speech=0.01 screening=0.99 silence=0.00 tone=0.00]
```

The WAV must be 8 kHz, 16-bit, mono, and start at the moment the call is answered; only the first 2.0 s is used. Convert with `sox in.wav -r 8000 -c 1 -b 16 out.wav`. Use `--model models/lite/amd.onnx` for the small model and `--human-threshold` to trade live people lost against recordings passed.

## What is in the repository

| Path | What |
|---|---|
| `models/full/`, `models/lite/` | trained models: `amd.onnx` (int16 PCM in, class probabilities out; the log-mel front end is inside the graph), `amd.json` (classes, input format), `report.json` (training history and test metrics) |
| `scripts/classify_wav.py` | classify WAV files locally |
| `amd/server.py` | WebSocket server: buffers 2.0 s per call, batches calls through ONNX, runs the tone detector, returns the verdict |
| `amd/tones.py` | streaming busy / reorder / SIT / ringback / dial / fax / beep detector |
| `client/amd-eagi.c` | Asterisk EAGI client, static binary, no dependencies |
| `scripts/` | data preparation, training, evaluation, ONNX export, load tests |
| `DATASET.md` | the dataset format and labelling rules, to build your own |

**Not included:** call recordings and transcripts. They are real phone calls and are not published. PyTorch checkpoints are not included either; the ONNX files are the trained models.

## Models

| Model | Params | Size | Held-out test, human threshold 0.4: live person lost / recording passed as HUMAN | Serve on |
|---|---|---|---|---|
| `models/full` | 584 k | 2.6 MB | 2.2 % / 1.1 % | GPU, or CPU up to about 1,000 concurrent calls |
| `models/lite` | 32 k | 0.4 MB | 3.4 % / 0.8 % | CPU |

Those test labels come from an earlier detector's verdicts and contain errors, so the rates are approximate and optimistic. See the agent-labelled result above for the hard cases.

## Capacity (measured, real-time streams, all calls deciding at the same instant)

| Setup | Concurrent calls | Connect → verdict p50 / p95 | Errors |
|---|---|---|---|
| full model, RTX 4090, 8 workers | 5,000 | 2.07 s / 2.29 s | 0 |
| lite model, CPU (Ryzen 9 9950X, 16 workers) | 5,000 | 1.94 s / 2.87 s | 0 |
| full model, CPU, 16 workers | 1,000 | 2.00 s / 2.58 s | 0 |

## Run the server

```bash
# GPU (needs onnxruntime-gpu: pip install "onnxruntime-gpu[cuda,cudnn]" numpy websockets)
python -m amd.server --model models/full/amd.onnx --device cuda --workers 8 \
    --host 0.0.0.0 --port 8085 --human-threshold 0.4

# CPU only (pip install onnxruntime numpy websockets)
python -m amd.server --model models/lite/amd.onnx --workers 16 --host 0.0.0.0 --port 8085 --human-threshold 0.4
```

Options: `--human-threshold` (default 0.4: lower = fewer live people lost, more recordings to agents), `--min-confidence` (below this, a non-human verdict becomes `UNKNOWN`), `--save-audio DIR` (keep every call's 2 s as `DIR/<day>/<cause>/<uniqueid>.wav` for review). Raise the open-file limit (`ulimit -n 65536`) for thousands of calls. The port has no authentication: firewall it to the Asterisk servers.

Protocol (one WebSocket per call): binary frames of slin16 from answer; optional text `{"type":"start","call_id":...}` first and `{"type":"close"}` on early hangup. Reply, once: `{"status":true,"amdstatus":"MACHINE","amdcause":"SCREENING","confidence":0.99,"p_human":0.01,"tone":"","audio_ms":2000,"elapsed_ms":2004}`.

## Asterisk side

```bash
gcc -O2 -static -o amd-eagi client/amd-eagi.c
cp amd-eagi /var/lib/asterisk/agi-bin/ && chmod 755 /var/lib/asterisk/agi-bin/amd-eagi
```

Dialplan, where the call is answered and `AMD()` would run:

```
same => n,EAGI(amd-eagi,ws://10.0.0.5:8085/,3000)
same => n,NoOp(AMD ${AMDSTATUS} ${AMDCAUSE})
```

Arguments: server URL (IP address, no DNS), overall timeout in ms (default 3000). On VICIdial, put it in place of the `AMD()` application in the AMD extension, before the AGI that reads `AMDSTATUS`/`AMDCAUSE`; check that your AGI accepts the new `AMDCAUSE` values.

## Load and integration tests

These need your own cached clips (see `DATASET.md`):

```bash
python scripts/replay.py --n 15000 --concurrency 5000 --realtime --procs 8   # server accuracy + load
python scripts/test_eagi.py --n 300 --parallel 100                            # C client, fake Asterisk
python scripts/eval_tones.py                                                  # tone detector
```

## Train your own

You need a manifest of labelled 8 kHz clips (see `DATASET.md`) and a CUDA GPU.

```bash
pip install -r requirements.txt
python scripts/resplit.py      --manifest data/manifest.csv --out data/v2split
python scripts/cache_audio.py  --split-dir data/v2split --out cache/v2split
python scripts/relabel.py      --src cache/v2split --dst cache/v2split_relabel
python scripts/train.py        --cache cache/v2split_relabel --out runs/baseline_relabel
python scripts/train.py        --cache cache/v2split_relabel --out runs/lite_v1 --arch lite --epochs 30
python scripts/export_onnx.py  --run runs/baseline_relabel --cache cache/v2split_relabel
```

## License

MIT. See `LICENSE`. Provided as is, with no warranty: you are responsible for how you use it on live calls, including compliance with the calling rules that apply to you.

For a hosted, supported detector for VICIdial, Asterisk, FreeSWITCH and Twilio, see [vmhunter.com](https://vmhunter.com).
