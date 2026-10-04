"""Stream test clips through the AMD server like live calls; check verdicts and measure capacity.

  --concurrency N   keep N calls open at once (each streams 100 ms frames)
  --realtime        pace frames at real time (2 s per call), as Asterisk would
Reports cause accuracy vs labels, the two business errors, and verdict latency
(time from the last audio frame sent to the verdict received).
"""
import argparse, asyncio, json, os, time
import numpy as np
import pandas as pd

FRAME = 800  # 100 ms


async def call(url, pcm, realtime, call_id):
    import websockets
    async with websockets.connect(url, max_size=2 ** 20, ping_interval=None, compression=None) as ws:
        await ws.send(json.dumps({"type": "start", "call_id": call_id}))
        t_start = time.monotonic()
        for i in range(0, len(pcm), FRAME):
            await ws.send(pcm[i:i + FRAME].tobytes())
            if realtime:
                await asyncio.sleep(max(0, t_start + (i + FRAME) / 8000 - time.monotonic()))
        t_last = time.monotonic()
        v = json.loads(await asyncio.wait_for(ws.recv(), 10))
        v["latency_ms"] = (time.monotonic() - t_last) * 1000
        return v


async def main_async(a, part=0, parts=1):
    df = pd.read_csv(os.path.join(a.cache, "test.csv"), dtype=str, keep_default_na=False)
    x = np.load(os.path.join(a.cache, "test.npy"), mmap_mode="r")
    keep = np.flatnonzero(((df.tier == "B") & df.coarse.isin(["human", "machine_speech", "screening", "silence", "tone"])).to_numpy())
    rng = np.random.default_rng(0)
    idx = rng.choice(keep, min(a.n, len(keep)), replace=False)[part::parts]
    sem = asyncio.Semaphore(max(1, a.concurrency // parts))
    results = [None] * len(idx)

    async def one(j, i):
        async with sem:
            try:
                results[j] = await call(a.url, np.asarray(x[i]), a.realtime, str(df.id.iloc[i]))
            except Exception as e:
                results[j] = {"error": repr(e)}

    t0 = time.monotonic()
    await asyncio.gather(*(one(j, i) for j, i in enumerate(idx)))
    wall = time.monotonic() - t0

    if parts > 1:
        return [(df.coarse.iloc[i], r) for i, r in zip(idx, results)], wall
    report([(df.coarse.iloc[i], r) for i, r in zip(idx, results)], wall)


def report(pairs, wall):
    ok = [(c, r) for c, r in pairs if r and "error" not in r]
    errs = [r for _, r in pairs if not r or "error" in r]
    want = {"human": "HUMAN", "machine_speech": "MACHINE", "screening": "SCREENING", "silence": "SILENCE", "tone": "TONE"}
    y = np.array([want[c] for c, _ in ok])
    got = np.array([r["amdcause"] for _, r in ok])
    lat = np.array([r["latency_ms"] for _, r in ok])
    is_h = y == "HUMAN"
    print(f"calls {len(pairs)}  ok {len(ok)}  errors {len(errs)}  wall {wall:.1f}s  ({len(ok) / wall:.0f} calls/s)")
    if errs:
        print("  first error:", errs[0])
    print(f"cause accuracy {np.mean(got == y):.4f} | live person lost {np.mean(got[is_h] != 'HUMAN'):.4f} "
          f"| recording -> HUMAN {np.mean(got[~is_h] == 'HUMAN'):.4f} | UNKNOWN {np.mean(got == 'UNKNOWN'):.4f}")
    print(f"verdict latency ms: p50 {np.percentile(lat, 50):.0f}  p95 {np.percentile(lat, 95):.0f}  p99 {np.percentile(lat, 99):.0f}  max {lat.max():.0f}")
    print("causes:", pd.Series(got).value_counts().to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8085/")
    ap.add_argument("--cache", default="cache/v2split_relabel")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=100)
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--procs", type=int, default=1, help="split the load over this many client processes")
    a = ap.parse_args()
    if a.procs == 1:
        asyncio.run(main_async(a))
        return
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(a.procs) as ex:
        res = list(ex.map(run_part, [(a, k, a.procs) for k in range(a.procs)]))
    report([p for r, _ in res for p in r], max(w for _, w in res))


def run_part(args):
    import resource
    resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))
    return asyncio.run(main_async(*args))


if __name__ == "__main__":
    main()
