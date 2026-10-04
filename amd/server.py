"""Acoustic AMD WebSocket server.

One WebSocket connection per answered call:
  client -> server: binary frames of raw slin16 (signed 16-bit little-endian, 8 kHz, mono),
                    from the moment of answer, any frame size (100 ms recommended)
                    optional text {"type": "start", "call_id": "..."} first
                    optional text {"type": "close"} if the call ends early
  server -> client: one text frame, the verdict, once 2.0 s of audio has arrived
                    (or on "close" / disconnect with what was received):
    {"status": true, "amdstatus": "HUMAN"|"MACHINE",
     "amdcause": "HUMAN"|"MACHINE"|"SCREENING"|"SILENCE"|"TONE"|"BUSY"|"UNKNOWN",
     "confidence": 0.97, "p_human": 0.97, "tone": "", "audio_ms": 2000, "elapsed_ms": 2013}

Decision at 2.0 s:
  1. busy / reorder / SIT / fax from the tone detector      -> BUSY or TONE
  2. p(human) >= --human-threshold                           -> HUMAN
  3. ringback / dial tone / beep from the tone detector      -> TONE
  4. most likely non-human class, if p >= --min-confidence   -> its cause
  5. otherwise                                               -> UNKNOWN
AMDSTATUS is HUMAN only when AMDCAUSE is HUMAN.

Run:  python -m amd.server --model runs/lite_v1/export/amd.onnx --port 8085 --workers 8
"""
import argparse, asyncio, json, logging, multiprocessing as mp, os, socket, time, wave
import numpy as np

from .tones import ToneDetector

WINDOW = 16000  # 2.0 s at 8 kHz
CAUSES = {"human": "HUMAN", "machine_speech": "MACHINE", "screening": "SCREENING", "silence": "SILENCE", "tone": "TONE"}
STRONG_TONES = {"busy", "reorder", "sit", "fax"}  # deterministic patterns: override the CNN
log = logging.getLogger("amd")


class Batcher:
    """Collects 2 s windows from all calls in this worker and runs them through ONNX together."""

    def __init__(self, model_path, threads, device="cpu", max_batch=256, max_wait_ms=5):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        if device == "cuda":  # needs the onnxruntime-gpu package
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
            providers = ["CUDAExecutionProvider"] + providers
        self.sess = ort.InferenceSession(model_path, so, providers=providers)
        if device == "cuda" and "CUDAExecutionProvider" not in self.sess.get_providers():
            raise RuntimeError("CUDA requested but onnxruntime could not use it")
        meta_path = os.path.join(os.path.dirname(model_path), "amd.json")
        self.classes = json.load(open(meta_path))["classes"]
        self.max_batch, self.max_wait = max_batch, max_wait_ms / 1000
        self.queue = asyncio.Queue()

    async def predict(self, pcm):
        fut = asyncio.get_running_loop().create_future()
        await self.queue.put((pcm, fut))
        return await fut

    async def run(self):
        loop = asyncio.get_running_loop()
        while True:
            items = [await self.queue.get()]
            deadline = loop.time() + self.max_wait
            while len(items) < self.max_batch:
                timeout = deadline - loop.time()
                if timeout <= 0:
                    break
                try:
                    items.append(await asyncio.wait_for(self.queue.get(), timeout))
                except asyncio.TimeoutError:
                    break
            x = np.stack([p for p, _ in items])
            try:
                probs = await loop.run_in_executor(None, lambda: self.sess.run(None, {"pcm": x})[0])
                for (_, fut), p in zip(items, probs):
                    if not fut.done():
                        fut.set_result(p)
            except Exception as e:  # never leave a call waiting
                for _, fut in items:
                    if not fut.done():
                        fut.set_exception(e)


def decide(probs, classes, tone, a):
    p = dict(zip(classes, map(float, probs)))
    tone_detail = tone[1] if tone else ""
    tone_kind = tone_detail.split(":")[0]
    if tone and tone_kind in STRONG_TONES:
        return tone[0], 1.0, tone_detail
    if p["human"] >= a.human_threshold:
        return "HUMAN", p["human"], tone_detail
    if tone:
        return tone[0], 1.0, tone_detail
    best = max((c for c in classes if c != "human"), key=p.get)
    if p[best] >= a.min_confidence:
        return CAUSES[best], p[best], tone_detail
    return "UNKNOWN", p[best], tone_detail


async def handle(ws, batcher, a):
    t0 = time.monotonic()
    buf = bytearray()
    det = ToneDetector()
    tone = None
    call_id = ""
    try:
        async for msg in ws:
            if isinstance(msg, bytes):
                if len(buf) % 2:  # keep sample alignment across odd-sized frames
                    msg = bytes(buf[-1:]) + msg
                    del buf[-1:]
                need = WINDOW * 2 - len(buf)
                chunk = msg[:need]
                buf += chunk
                if chunk and not tone:
                    tone = det.push(np.frombuffer(chunk[: len(chunk) // 2 * 2], np.int16))
                if len(buf) >= WINDOW * 2:
                    break
            else:
                try:
                    m = json.loads(msg)
                except ValueError:
                    continue
                if m.get("type") == "start":
                    call_id = str(m.get("call_id", ""))[:64]
                elif m.get("type") == "close":
                    break
    except Exception:
        pass  # disconnect: decide with what arrived, if the socket can still take the reply

    pcm = np.zeros(WINDOW, np.int16)
    got = np.frombuffer(bytes(buf[: len(buf) // 2 * 2]), np.int16)
    pcm[: len(got)] = got
    try:
        probs = await batcher.predict(pcm)
        cause, conf, tone_detail = decide(probs, batcher.classes, tone, a)
    except Exception as e:
        log.error("inference failed: %s", e)
        probs, cause, conf, tone_detail = None, "UNKNOWN", 0.0, ""
    verdict = {
        "status": True,
        "amdstatus": "HUMAN" if cause == "HUMAN" else "MACHINE",
        "amdcause": cause,
        "confidence": round(conf, 4),
        "p_human": round(float(probs[batcher.classes.index("human")]), 4) if probs is not None else None,
        "tone": tone_detail,
        "audio_ms": len(got) * 1000 // 8000,
        "elapsed_ms": int((time.monotonic() - t0) * 1000),
    }
    try:
        await ws.send(json.dumps(verdict))
    except Exception:
        pass
    log.info(json.dumps({"call_id": call_id, **verdict,
                         "probs": [round(float(v), 4) for v in probs] if probs is not None else None}))
    if a.save_audio and call_id:
        save(a.save_audio, call_id, cause, got)


def save(root, call_id, cause, pcm):
    day = time.strftime("%Y%m%d")
    d = os.path.join(root, day, cause)
    os.makedirs(d, exist_ok=True)
    safe = "".join(ch for ch in call_id if ch.isalnum() or ch in "-_.")
    with wave.open(os.path.join(d, f"{safe}.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
        w.writeframes(pcm.tobytes())


async def serve(sock, a):
    import websockets
    batcher = Batcher(a.model, a.threads, a.device)
    asyncio.create_task(batcher.run())
    async with websockets.serve(lambda ws: handle(ws, batcher, a), sock=sock, max_size=2 ** 20,
                                ping_interval=None, compression=None, backlog=4096):
        await asyncio.Future()


def worker(sock, a):
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s w{os.getpid()} %(message)s")
    asyncio.run(serve(sock, a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="amd.onnx (amd.json beside it)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8085)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 4))
    ap.add_argument("--threads", type=int, default=1, help="onnxruntime threads per worker")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--human-threshold", type=float, default=0.4)
    ap.add_argument("--min-confidence", type=float, default=0.5)
    ap.add_argument("--save-audio", default="", help="dir to keep each call's 2 s as WAV (for review)")
    a = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((a.host, a.port))
    sock.listen(4096)
    sock.set_inheritable(True)
    procs = [mp.get_context("fork").Process(target=worker, args=(sock, a), daemon=True) for _ in range(a.workers)]
    for p in procs:
        p.start()
    print(f"amd server on {a.host}:{a.port}, {a.workers} workers on {a.device}, "
          f"model {a.model}, human threshold {a.human_threshold}", flush=True)
    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
