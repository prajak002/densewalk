# Which box is faster — measured 2026-09-19

Identical benchmark (`bench.py`) on both. BLAS pinned to 1 thread/worker.

|                              | vast RTX 5060 Ti | AWS L4        | ratio      |
|------------------------------|------------------|---------------|------------|
| GPU fp32 burst (TFLOPS)      | 17.51            | 11.05         | 1.58x      |
| GPU fp32 sustained 20s       | **17.13** (-2%)  | 10.19 (-8%)   | **1.68x**  |
| GPU bandwidth (GB/s)         | **386.3**        | 231.4         | 1.67x      |
| CPU 1 core (tasks/s)         | 18.9             | 17.7          | 1.07x      |
| CPU best all-core (tasks/s)  | **229.0** (32p)  | 40.1 (4p)     | **5.71x**  |
| ffmpeg 361-frame extract     | 0.36 s           | no ffmpeg     | -          |
| $/hr                         | **0.364**        | 1.0064        | 2.76x cheaper |

Vast wins every axis. GPU 1.7x (the L4's 72W cap shows as -8% sustained droop
vs -2%), CPU 5.7x, and it costs 2.8x less.

Caveat found: `cpu_cores_effective` on the vast container is **21.3**, not the
128 `nproc` reports - it is CPU-quota'd. 32 processes beat 128 (229 vs 66
tasks/s), so COLMAP must be capped near 24 threads, not 128.

COLMAP 3.9.1 from apt is built **without CUDA**, so SfM is CPU SIFT - which is
why the 5.7x CPU gap, not the GPU gap, is the one that decides this.
