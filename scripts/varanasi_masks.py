"""M1-Varanasi step 2: segment the dynamic crowd out of every frame.

AV2 gave us ground-truth cuboids to prompt SAM2 with (scripts/mask_dynamics.py).
A phone video gives us nothing, so dynamics come from an instance segmenter
over the COCO dynamic classes -- people, two-wheelers, vehicles, animals.

Writes TWO mask sets, because the two consumers use opposite conventions:
  masks_train/<name>.png    255 = dynamic  (excluded from the gsplat loss,
                                            same convention as M1/AV2)
  masks_colmap/<name>.png   0   = ignore   (COLMAP's ImageReader.mask_path
                                            convention: zero pixels are dropped,
                                            so this set is the inverse)

Masking before SfM is the point: features on a moving pedestrian are
consistent across neighbouring frames and COLMAP will happily triangulate
them into the static model, which corrupts the poses for every later step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# COCO ids that move in a street scene. Parked vehicles are a real risk of
# over-masking (see the AV2 lesson) but in this clip the road traffic is
# moving, and a wrongly-kept moving bike is far more damaging to SfM than a
# wrongly-dropped parked one.
DYNAMIC_COCO_IDS = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus",
    7: "truck", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--weights", default="yolo11x-seg.pt")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="low on purpose: a missed person costs more than a false one")
    ap.add_argument("--dilate-px", type=int, default=11,
                    help="grow masks to catch soft edges, shadows and motion blur")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640,
                    help="inference resolution. The source is 640x360; running "
                         "inference at 1280 upsamples it, which is what recovers "
                         "the distant pedestrians a 640 pass misses entirely.")
    args = ap.parse_args()

    from ultralytics import YOLO

    train_dir = args.out_dir / "masks_train"
    colmap_dir = args.out_dir / "masks_colmap"
    train_dir.mkdir(parents=True, exist_ok=True)
    colmap_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.images.glob("*.jpg"))
    if not paths:
        raise SystemExit(f"no .jpg under {args.images}")
    print(f"{len(paths)} images, weights={args.weights}, conf={args.conf}")

    model = YOLO(args.weights)
    kernel = np.ones((args.dilate_px, args.dilate_px), np.uint8)

    stats = []
    for i in range(0, len(paths), args.batch):
        chunk = paths[i:i + args.batch]
        for path, result in zip(chunk, model.predict(
                [str(p) for p in chunk], conf=args.conf, imgsz=args.imgsz,
                verbose=False)):
            img_h, img_w = result.orig_shape
            dynamic = np.zeros((img_h, img_w), np.uint8)
            n_inst = 0
            if result.masks is not None:
                cls = result.boxes.cls.cpu().numpy().astype(int)
                data = result.masks.data.cpu().numpy()  # (N, mh, mw), model res
                for c, m in zip(cls, data):
                    if c not in DYNAMIC_COCO_IDS:
                        continue
                    m = cv2.resize(m, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
                    dynamic |= (m > 0.5).astype(np.uint8) * 255
                    n_inst += 1
            if args.dilate_px > 1:
                dynamic = cv2.dilate(dynamic, kernel)

            cv2.imwrite(str(train_dir / f"{path.name}.png"), dynamic)
            cv2.imwrite(str(colmap_dir / f"{path.name}.png"),
                        np.where(dynamic > 0, 0, 255).astype(np.uint8))
            stats.append({"name": path.name, "instances": n_inst,
                          "dynamic_frac": float((dynamic > 0).mean())})
        print(f"  {min(i + args.batch, len(paths))}/{len(paths)}", flush=True)

    fracs = np.array([s["dynamic_frac"] for s in stats])
    summary = {
        "n_images": len(stats),
        "mean_dynamic_frac": float(fracs.mean()),
        "max_dynamic_frac": float(fracs.max()),
        "frames_over_50pct_dynamic": int((fracs > 0.5).sum()),
        "mean_instances": float(np.mean([s["instances"] for s in stats])),
    }
    (args.out_dir / "masks_report.json").write_text(
        json.dumps({"summary": summary, "frames": stats}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
