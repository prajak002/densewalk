"""Identical benchmark for both boxes, measuring what actually gates M1:
GPU fp32 + bandwidth (gsplat rasterisation), single-core and all-core CPU
(COLMAP CPU SIFT matching / bundle adjustment), and a real ffmpeg stage.

BLAS threading is pinned to 1 per worker: without that, one process per core
oversubscribes a 128-core box and the many-core machine benchmarks SLOWER
than a 4-core one, which is an artefact, not a result.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"

import json, subprocess, sys, time, multiprocessing as mp
import numpy as np
import torch

def emit(k, v):
    print(f"  {k}: {v}", flush=True)
    res[k] = v

res = {}
print(f"HOST {os.uname().nodename} | {torch.cuda.get_device_name(0)} | "
      f"torch {torch.__version__} | {os.cpu_count()} cores", flush=True)
res.update(host=os.uname().nodename, gpu=torch.cuda.get_device_name(0),
           torch=torch.__version__, cores=os.cpu_count())

torch.backends.cuda.matmul.allow_tf32 = False
n = 8192
a = torch.randn(n, n, device="cuda"); b = torch.randn(n, n, device="cuda")
for _ in range(3): a @ b
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(20): a @ b
torch.cuda.synchronize()
emit("gpu_fp32_tflops", round(20 * 2 * n**3 / (time.perf_counter() - t) / 1e12, 2))

x = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
y = torch.empty_like(x)
for _ in range(3): y.copy_(x)
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(30): y.copy_(x)
torch.cuda.synchronize()
emit("gpu_bandwidth_gbs", round(30 * 2 * x.nbytes / (time.perf_counter() - t) / 1e9, 1))

# Sustained: a 72W-capped card drops here, a 180W desktop card does not.
torch.cuda.synchronize(); t = time.perf_counter(); i = 0
while time.perf_counter() - t < 20: a @ b; i += 1
torch.cuda.synchronize()
emit("gpu_fp32_tflops_sustained20s", round(i * 2 * n**3 / (time.perf_counter() - t) / 1e12, 2))
del a, b, x, y; torch.cuda.empty_cache()

def qr_task(_):
    m = np.random.rand(400, 400)
    for _ in range(8): m = np.linalg.qr(m)[0]
    return float(m[0, 0])

t = time.perf_counter(); qr_task(0)
emit("cpu_1core_task_s", round(time.perf_counter() - t, 3))

N = 512
for nproc in sorted({1, 4, min(os.cpu_count(), 32), os.cpu_count()}):
    t = time.perf_counter()
    with mp.Pool(nproc) as p: p.map(qr_task, range(N))
    dt = time.perf_counter() - t
    emit(f"cpu_{nproc}proc_{N}tasks_s", round(dt, 2))
    emit(f"cpu_{nproc}proc_tasks_per_s", round(N / dt, 1))

vid, out = "/tmp/varanasi1.mp4", "/tmp/bench_frames"
if os.path.exists(vid):
    subprocess.run(["rm", "-rf", out], check=False); os.makedirs(out, exist_ok=True)
    t = time.perf_counter()
    subprocess.run(["ffmpeg", "-v", "error", "-i", vid, "-q:v", "2",
                    f"{out}/f%05d.jpg", "-y"], check=True)
    emit("ffmpeg_extract_493f_s", round(time.perf_counter() - t, 2))
    emit("frames_written", len(os.listdir(out)))

print("JSON " + json.dumps(res), flush=True)
