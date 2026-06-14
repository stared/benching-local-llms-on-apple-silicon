#!/usr/bin/env python3
"""
Apples-to-apples benchmark for local 8-bit Qwen3.6 models on Apple Silicon,
comparing llama.cpp (GGUF) vs MLX through their OpenAI-compatible servers.

For each (engine, model) it:
  1. kills any stray model server,
  2. launches the server itself (so exactly one model is ever resident),
  3. measures cold load time (process start -> first token served),
  4. runs N timed trials at each prompt size, streaming, measuring:
       - TTFT      : time to first token (s)
       - prefill   : prompt tokens / TTFT (tok/s)   [prompt-processing speed]
       - decode    : generated tokens / generation time (tok/s) [the headline number]
  5. samples memory the whole time:
       - rss_peak  : server process RSS (GB)
       - sys_peak  : system-wide "used" memory delta vs baseline (GB)  [catches GPU buffers]
  6. tears the server down and reports medians as a markdown table + JSON.

Stdlib only — no pip deps. Run:  python3 bench.py
"""
import argparse, json, os, re, shutil, signal, subprocess, sys, time, urllib.request, urllib.error
from threading import Thread, Event

HOME = os.path.expanduser("~")
HOST = "127.0.0.1"
PORT = 8099                      # dedicated bench port, avoids your 8080/8081 servers
LOGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_logs")
DS4_DIR = f"{HOME}/not-my-repos/ds4"   # ds4-server runs from here (resolves ds4flash.gguf)

# ---- what to benchmark -------------------------------------------------------
# ds4 is a DIFFERENT (91 GB) model on a different engine — including it is a
# whole-setup comparison, not engine-isolated. Opt in with --only ds4 so a normal
# run doesn't load 91 GB. Measured with the identical trial()/sweep as the rest.
# MTP speculative-decoding flags (llama.cpp). The ~/models GGUFs are the unsloth
# -MTP-GGUF builds, which run as a plain model WITHOUT these flags (= baseline)
# and enable speculative decoding WITH them — so the same file gives both A/B sides.
# Sweep the draft depth with --mtp-nmax (1-6); blog's M1 optimum was 3.
MTP_ARGS = ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]

CONFIGS = [
    {"name": "llama.cpp 35B-A3B Q8_0", "engine": "llama",
     "model": f"{HOME}/models/Qwen3.6-35B-A3B-Q8_0.gguf"},
    {"name": "llama.cpp 27B Q8_0",     "engine": "llama",
     "model": f"{HOME}/models/Qwen3.6-27B-Q8_0.gguf"},
    {"name": "llama.cpp 35B-A3B Q8_0 +MTP", "engine": "llama",
     "model": f"{HOME}/models/Qwen3.6-35B-A3B-Q8_0.gguf", "extra_args": MTP_ARGS},
    {"name": "llama.cpp 27B Q8_0 +MTP", "engine": "llama",
     "model": f"{HOME}/models/Qwen3.6-27B-Q8_0.gguf", "extra_args": MTP_ARGS},
    {"name": "MLX 35B-A3B 8bit",       "engine": "mlx",
     "model": "mlx-community/Qwen3.6-35B-A3B-8bit"},
    {"name": "MLX 27B 8bit",           "engine": "mlx",
     "model": "mlx-community/Qwen3.6-27B-8bit"},
    {"name": "ds4 DeepSeek-V4-Flash q2-q4", "engine": "ds4",
     "model": "ds4flash.gguf"},
]

# ---- memory helpers (system-wide, captures unified GPU buffers) ---------------
def _pagesize():
    return int(subprocess.check_output(["sysctl", "-n", "hw.pagesize"]).strip())

PAGE = _pagesize()

def sys_used_gb():
    """System memory in active+wired+compressed (i.e. not free/purgeable), in GB."""
    out = subprocess.check_output(["vm_stat"]).decode()
    g = {k.strip(): int(v) for k, v in re.findall(r'"?([A-Za-z ]+)"?:\s+(\d+)\.', out)}
    used_pages = (g.get("Pages active", 0) + g.get("Pages wired down", 0)
                  + g.get("Pages occupied by compressor", 0))
    return used_pages * PAGE / 1e9

def rss_gb(pid):
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)]).strip()
        return int(out) / 1e6  # ps gives KB
    except Exception:
        return 0.0

class MemSampler(Thread):
    """Polls process RSS and system-used memory until stopped; keeps the peak."""
    def __init__(self, pid, interval=0.25):
        super().__init__(daemon=True)
        self.pid, self.interval = pid, interval
        self.stop = Event()
        self.rss_peak = 0.0
        self.sys_peak = 0.0
    def run(self):
        while not self.stop.is_set():
            self.rss_peak = max(self.rss_peak, rss_gb(self.pid))
            self.sys_peak = max(self.sys_peak, sys_used_gb())
            time.sleep(self.interval)

# ---- server lifecycle --------------------------------------------------------
def kill_stray():
    for name in ("llama-server", "mlx_lm.server", "mlx_lm", "ds4-server"):
        subprocess.run(["pkill", "-f", name], stderr=subprocess.DEVNULL)
    time.sleep(1.0)

def launch(cfg):
    os.makedirs(LOGDIR, exist_ok=True)
    log = open(os.path.join(LOGDIR, cfg["engine"] + "_" +
                            re.sub(r"\W+", "_", cfg["name"]) + ".log"), "w")
    cwd = None
    if cfg["engine"] == "llama":
        bin_ = shutil.which("llama-server")
        cmd = [bin_, "-m", cfg["model"], "-ngl", "999", "-fa", "on",
               "-c", "16384", "--parallel", "1", "--host", HOST, "--port", str(PORT)]
    elif cfg["engine"] == "ds4":
        bin_ = os.path.join(DS4_DIR, "ds4-server")
        cwd = DS4_DIR                       # so "ds4flash.gguf" symlink resolves
        cmd = [bin_, "-m", cfg["model"], "--metal", "-c", "16384",
               "--host", HOST, "--port", str(PORT),
               "--kv-disk-dir", "/tmp/ds4-kv", "--kv-disk-space-mb", "8192"]
    else:
        bin_ = shutil.which("mlx_lm.server")
        cmd = [bin_, "--model", cfg["model"], "--host", HOST, "--port", str(PORT),
               "--max-tokens", "4096"]
    cmd += cfg.get("extra_args") or []          # e.g. MTP speculative-decoding flags
    if not bin_ or (cwd and not os.path.exists(bin_)):
        raise SystemExit(f"binary for {cfg['engine']} not found: {bin_}")
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=cwd)
    return p, log

URL = f"http://{HOST}:{PORT}/v1/chat/completions"
_REQ = [0]   # monotonic request counter -> unique prompt prefix defeats KV cache

def _post(model_name, messages, max_tokens, stream, timeout, extra=None):
    body = {
        "model": model_name, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0.0, "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    if extra:
        body.update(extra)
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)

def wait_ready(proc, model_name, timeout=900):
    """Poll with a tiny request until the model serves a token. Returns load seconds."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise SystemExit("server exited during load — check bench_logs/")
        try:
            r = _post(model_name, [{"role": "user", "content": "hi"}], 1, False, 30)
            r.read()
            return time.time() - t0
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(1.0)
    raise SystemExit("server did not become ready in time")

# ---- one trial ---------------------------------------------------------------
# "A token is a token": we measure TOTAL throughput (reasoning + answer), so we
# never split content from thinking. Non-streaming avoids fragile per-engine SSE
# field parsing. A 1-token request isolates prefill; a full request gives
# prefill+decode; subtracting yields total decode tok/s over ALL generated tokens.
def _complete(model_name, prompt, max_tokens):
    """One non-streaming completion. Returns (wall_seconds, usage_dict)."""
    msgs = [{"role": "user", "content": prompt}]
    t0 = time.time()
    # explicitly disable llama.cpp prompt cache (MLX/ds4 ignore the field)
    resp = _post(model_name, msgs, max_tokens, False, 600, extra={"cache_prompt": False})
    usage = json.loads(resp.read().decode("utf-8", "ignore")).get("usage") or {}
    return time.time() - t0, usage

def trial(model_name, prompt, gen_tokens):
    # unique prefix per request => neither engine can reuse a cached prefill
    _REQ[0] += 1
    t_pre, u1 = _complete(model_name, f"[req {_REQ[0]}] " + prompt, 1)
    ptoks = u1.get("prompt_tokens") or max(1, len(prompt) // 4)
    _REQ[0] += 1
    t_full, u2 = _complete(model_name, f"[req {_REQ[0]}] " + prompt, gen_tokens)
    gen = u2.get("completion_tokens") or gen_tokens     # every token, reasoning included
    decode_time = max(1e-6, t_full - t_pre)             # subtract prefill -> pure decode
    return {
        "ttft": t_pre,
        "prefill_tps": ptoks / max(1e-6, t_pre),
        "decode_tps": (gen - 1) / decode_time if gen > 1 else 0.0,
        "gen_tokens": gen, "prompt_tokens": ptoks,
        "usage_exact": bool(u2),
    }

def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2

# ---- prompts -----------------------------------------------------------------
# Build prompts at target token sizes (~4 chars/token) to sweep context length.
# Actual prompt_tokens come back from each server's usage and are reported as-is.
_FILLER = ("def process(data):\n    result = []\n    for row in data:\n"
           "        result.append(transform(row))\n    return result\n")

def make_prompt(target_tokens):
    head = "Here is some code. Summarize what it does and suggest improvements.\n\n"
    tail = "\n\nNow give a concise analysis."
    need = max(0, target_tokens * 4 - len(head) - len(tail))
    body = (_FILLER * (need // len(_FILLER) + 1))[:need]
    return head + body + tail

def label_for(t):
    return f"{t//1024}k" if t >= 1024 else f"{t}tok"

# ---- main --------------------------------------------------------------------
def run(cfg, runs, gen_tokens, prompts):
    print(f"\n=== {cfg['name']} ===", flush=True)
    kill_stray()
    base_sys = sys_used_gb()
    # Some servers validate the request's model field against what they loaded
    # (MLX, ds4); llama-server ignores it. Send the right id where it matters.
    model_name = {"mlx": cfg["model"], "ds4": "deepseek-v4-flash"}.get(
        cfg["engine"], "bench")
    proc, log = launch(cfg)
    sampler = MemSampler(proc.pid)
    sampler.start()
    try:
        load_s = wait_ready(proc, model_name)
        print(f"  cold load: {load_s:.1f}s", flush=True)
        # warmup once at the largest prompt (compiles Metal kernels, warms KV path)
        trial(model_name, max(prompts.values(), key=len), 32)
        result = {"name": cfg["name"], "engine": cfg["engine"],
                  "load_s": round(load_s, 1), "sizes": {}}
        for label, prompt in prompts.items():
            rows = [trial(model_name, prompt, gen_tokens) for _ in range(runs)]
            result["sizes"][label] = {
                "prompt_tokens": rows[0]["prompt_tokens"],
                "ttft_s":      round(median([r["ttft"] for r in rows]), 3),
                "prefill_tps": round(median([r["prefill_tps"] for r in rows]), 1),
                "decode_tps":  round(median([r["decode_tps"] for r in rows]), 1),
                "usage_exact": rows[0]["usage_exact"],
            }
            s = result["sizes"][label]
            print(f"  {label:6s} prefill {s['prefill_tps']:7.1f} t/s | "
                  f"decode {s['decode_tps']:6.1f} t/s | ttft {s['ttft_s']:.3f}s "
                  f"({s['prompt_tokens']} prompt tok)", flush=True)
    finally:
        sampler.stop.set()
        sampler.join(timeout=2)
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    result["rss_peak_gb"] = round(sampler.rss_peak, 1)
    result["sys_delta_gb"] = round(sampler.sys_peak - base_sys, 1)
    print(f"  mem: rss_peak {result['rss_peak_gb']} GB | "
          f"system delta {result['sys_delta_gb']} GB", flush=True)
    return result

def markdown(results, gen_tokens):
    out = [f"# Benchmark results ({gen_tokens} gen tokens, median of trials)\n"]
    out.append("| model | prompt | prefill t/s | decode t/s | TTFT s | "
               "RSS GB | sys ΔGB | load s |")
    out.append("|---|---|--:|--:|--:|--:|--:|--:|")
    for r in results:
        for label, s in r["sizes"].items():
            out.append(f"| {r['name']} | {label} "
                       f"({s['prompt_tokens']} tok) | {s['prefill_tps']} | "
                       f"**{s['decode_tps']}** | {s['ttft_s']} | "
                       f"{r['rss_peak_gb']} | {r['sys_delta_gb']} | {r['load_s']} |")
    exact = all(s["usage_exact"] for r in results for s in r["sizes"].values())
    if not exact:
        out.append("\n> ⚠️ token counts approximated from stream chunks for some rows "
                   "(server did not return usage). Decode t/s still reliable.")
    return "\n".join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3, help="timed trials per prompt size")
    ap.add_argument("--gen-tokens", type=int, default=400, help="tokens to generate")
    ap.add_argument("--prompt-tokens", default="128,1024,4096,8192",
                    help="comma-separated context sizes to sweep")
    ap.add_argument("--only", default="", help="substring filter on config name")
    ap.add_argument("--mtp-nmax", type=int, default=None,
                    help="override --spec-draft-n-max on +MTP configs (sweep 1-6)")
    args = ap.parse_args()

    if args.mtp_nmax is not None:                # rewrite draft depth on MTP configs
        for c in CONFIGS:
            ea = c.get("extra_args")
            if ea and "--spec-draft-n-max" in ea:
                ea = list(ea)
                ea[ea.index("--spec-draft-n-max") + 1] = str(args.mtp_nmax)
                c["extra_args"] = ea

    cfgs = [c for c in CONFIGS if args.only.lower() in c["name"].lower()]
    if not cfgs:
        raise SystemExit("no configs match --only")

    sizes = [int(x) for x in args.prompt_tokens.split(",")]
    prompts = {label_for(t): make_prompt(t) for t in sizes}

    print(f"machine: {subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string']).decode().strip()}")
    print(f"sweep: context {sizes} tok x {args.gen_tokens} gen x {args.runs} runs")
    print(f"baseline system used: {sys_used_gb():.1f} GB  "
          f"(close heavy apps first — see NOTES.md)")
    results = [run(c, args.runs, args.gen_tokens, prompts) for c in cfgs]

    md = markdown(results, args.gen_tokens)
    with open("bench_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("bench_results.md", "w") as f:
        f.write(md + "\n")
    print("\n" + md)
    print("\nsaved -> bench_results.json, bench_results.md")

if __name__ == "__main__":
    main()
