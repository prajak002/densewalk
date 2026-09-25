"""x4 super-resolution of the Varanasi frames (640x360 -> 2560x1440) with Swin2SR.

Model: caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr (Apache-2.0), the real-world
degradation variant -- the phone clip is compressed and slightly blurred, which the
classical (bicubic-trained) variant handles worse.

The processor pads right/bottom to a multiple of 8; the output is cropped back to
exactly 4x the input so pixel coordinates stay a clean x4 of the original, which
the sprite re-cut relies on (boxes and masks scale by exactly 4).
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

ap = argparse.ArgumentParser()
ap.add_argument("--images", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--model", default="caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--jpeg-quality", type=int, default=94)
a = ap.parse_args()

proc = AutoImageProcessor.from_pretrained(a.model)
model = Swin2SRForImageSuperResolution.from_pretrained(a.model).to("cuda").eval().half()
a.out.mkdir(parents=True, exist_ok=True)
paths = sorted(a.images.glob("*.jpg"))[: a.limit or None]
t0 = time.time()
for i, p in enumerate(paths):
    if (a.out / p.name).exists():
        continue
    bgr = cv2.imread(str(p)); H, W = bgr.shape[:2]
    x = proc(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), return_tensors="pt")["pixel_values"].to("cuda").half()
    with torch.no_grad():
        y = model(pixel_values=x).reconstruction[0].float().clamp(0, 1)
    y = y[:, : H * 4, : W * 4].permute(1, 2, 0).cpu().numpy()
    cv2.imwrite(str(a.out / p.name), cv2.cvtColor((y * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, a.jpeg_quality])
    if i % 50 == 0:
        print(f"  {i}/{len(paths)}  {time.time() - t0:.0f}s", flush=True)
print(f"done {len(paths)} frames in {time.time() - t0:.0f}s -> {a.out}")
