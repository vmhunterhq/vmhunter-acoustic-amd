"""Run client/amd-eagi like Asterisk would, without Asterisk.

For each test clip: start the binary with the AGI environment on stdin, stream the clip
on fd 3 at real time (20 ms writes, like EAGI), answer its AGI commands with
"200 result=1", and collect the AMDSTATUS / AMDCAUSE it sets.
Also checks the failure paths: server down, and a timeout.
"""
import argparse, fcntl, os, re, subprocess, threading, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd


def run_eagi(binary, url, pcm, timeout_ms=3000, uniqueid="1700000000.1", stream_s=None):
    r0, w = os.pipe()
    r = fcntl.fcntl(r0, fcntl.F_DUPFD, 10)  # never fd 3 itself, or the redirect below would close it
    os.close(r0)
    # Asterisk hands EAGI audio over on fd 3; a shell redirect puts our pipe there
    p = subprocess.Popen(["bash", "-c", f'exec "$0" "$1" "$2" 3<&{r} {r}<&-', binary, url, str(timeout_ms)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, pass_fds=(r,), text=True)
    os.close(r)
    p.stdin.write(f"agi_request: amd-eagi\nagi_channel: SIP/test-0001\nagi_uniqueid: {uniqueid}\n\n")
    p.stdin.flush()

    def feed():  # Asterisk writes 20 ms of audio every 20 ms and keeps going after 2 s
        t0 = time.monotonic()
        data = pcm.tobytes()
        limit = len(data) if stream_s is None else int(stream_s * 16000)
        try:
            for i in range(0, limit, 320):
                os.write(w, data[i:i + 320])
                time.sleep(max(0, t0 + (i + 320) / 16000 - time.monotonic()))
            while p.poll() is None and stream_s is None:  # silence after the clip
                os.write(w, b"\0" * 320); time.sleep(0.02)
        except OSError:
            pass
        finally:
            os.close(w)

    threading.Thread(target=feed, daemon=True).start()
    vars_, t0 = {}, time.monotonic()
    for line in p.stdout:
        m = re.match(r'SET VARIABLE (\w+) "(.*)"', line.strip())
        if m:
            vars_[m.group(1)] = m.group(2)
        p.stdin.write("200 result=1\n"); p.stdin.flush()
    p.wait()
    vars_["elapsed_ms"] = int((time.monotonic() - t0) * 1000)
    return vars_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="client/amd-eagi")
    ap.add_argument("--url", default="ws://127.0.0.1:8085/")
    ap.add_argument("--cache", default="cache/v2split_relabel")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--parallel", type=int, default=50)
    a = ap.parse_args()

    df = pd.read_csv(os.path.join(a.cache, "test.csv"), dtype=str, keep_default_na=False)
    x = np.load(os.path.join(a.cache, "test.npy"), mmap_mode="r")
    keep = np.flatnonzero(((df.tier == "B") & df.coarse.isin(["human", "machine_speech", "screening", "silence", "tone"])).to_numpy())
    idx = np.random.default_rng(1).choice(keep, a.n, replace=False)
    with ThreadPoolExecutor(a.parallel) as ex:
        res = list(ex.map(lambda i: run_eagi(a.binary, a.url, np.asarray(x[i]), uniqueid=df.id.iloc[i]), idx))
    want = {"human": "HUMAN", "machine_speech": "MACHINE", "screening": "SCREENING", "silence": "SILENCE", "tone": "TONE"}
    y = np.array([want[c] for c in df.coarse.iloc[idx]])
    got = np.array([r.get("AMDCAUSE", "") for r in res])
    st = np.array([r.get("AMDSTATUS", "") for r in res])
    el = np.array([r["elapsed_ms"] for r in res])
    print(f"{a.n} calls via EAGI client: cause accuracy {np.mean(got == y):.3f}, "
          f"status ok {np.mean((st == 'HUMAN') == (y == 'HUMAN')):.3f}, "
          f"rule held (HUMAN iff cause HUMAN) {np.all((st == 'HUMAN') == (got == 'HUMAN'))}, "
          f"time from answer: p50 {np.percentile(el, 50):.0f} ms, max {el.max()} ms")
    print("causes:", pd.Series(got).value_counts().to_dict())

    pcm = np.asarray(x[idx[0]])
    print("server down  ->", run_eagi(a.binary, "ws://127.0.0.1:1/", pcm))
    print("bad url      ->", run_eagi(a.binary, "ws://not-an-ip:8085/", pcm))
    print("timeout 800ms->", run_eagi(a.binary, a.url, pcm, timeout_ms=800))
    print("hangup at 1s ->", run_eagi(a.binary, a.url, pcm, stream_s=1.0))


if __name__ == "__main__":
    main()
