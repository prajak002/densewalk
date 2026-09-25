"""Parallel plain-HTTPS fetch of many small HF dataset files; skips files already present.
snapshot_download managed ~2.7 files/s on 13.7k keyframes; this uses 16 connections."""
import os, sys, time, fnmatch, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi
REPO, OUT = "s-alam/dense_walk_mini", "/workspace/dw_data"
TOK = os.environ["HF_TOKEN"]
RUN = "dca_results/cosmos3-nano_v2/run_20260820_000205"
PATS = ["json_openvla/*", "keyframes/*/*",
        f"{RUN}/V2_full/dca_full/adapter_best/*", f"{RUN}/V2_full/dca_full/heads_best.pt",
        f"{RUN}/V2_full/lora_only/adapter_best/*", f"{RUN}/V2_full/lora_only/heads_best.pt",
        f"{RUN}/predictions_V2_full_dca_full.json", f"{RUN}/predictions_V2_full_lora_only.json"]
files = [f for f in HfApi(token=TOK).list_repo_files(REPO, repo_type="dataset")
         if any(fnmatch.fnmatch(f, p) for p in PATS)]
todo = [f for f in files if not os.path.exists(os.path.join(OUT, f))]
print(f"{len(files)} wanted, {len(todo)} missing", flush=True)
S = requests.Session(); S.headers["Authorization"] = f"Bearer {TOK}"
def get(f):
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{f}"
    for k in range(8):
        try:
            r = S.get(url, timeout=120)
            if r.status_code == 200:
                dst = os.path.join(OUT, f); os.makedirs(os.path.dirname(dst), exist_ok=True)
                tmp = dst + ".part"; open(tmp, "wb").write(r.content); os.replace(tmp, dst); return True
            if r.status_code not in (429, 500, 502, 503, 504): raise RuntimeError(f"{r.status_code} {f}")
        except requests.RequestException: pass
        time.sleep(min(60, 2 ** k))
    raise RuntimeError(f"gave up on {f}")
t0, done = time.time(), 0
with ThreadPoolExecutor(16) as ex:
    for fu in as_completed([ex.submit(get, f) for f in todo]):
        fu.result(); done += 1
        if done % 1000 == 0: print(f"  {done}/{len(todo)}  {done/(time.time()-t0):.1f} files/s", flush=True)
miss = [f for f in files if not os.path.exists(os.path.join(OUT, f))]
print("still missing:", len(miss)); sys.exit(1 if miss else 0)
