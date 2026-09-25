"""Step 4: mask people and vehicles out of AV2 ring-camera frames.

SAM2 is promptable, not semantic -- on its own it cannot be told "people
and vehicles". Rather than bolt on a separate detector, we prompt it with
AV2's own ground-truth 3D cuboids for the dynamic categories, projected
into each image. Ground-truth boxes beat detected boxes, and the data is
already on disk.

Writes, per camera:
  masks/<cam>/<timestamp>.png   255 = dynamic (exclude), 0 = background
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from densewalk import frames
from densewalk.av2 import Av2Sequence, RING_CAMERAS, assert_av2_frame_sane


def cuboid_to_box_prompt(seq, cam, row, world_to_cam):
    """Project one cuboid's 8 corners into the image, return an xyxy box
    (or None if it is behind the camera / off-image)."""
    corners_ego = Av2Sequence.cuboid_corners_ego(row)
    ego_pose = seq.city_se3_ego(int(row.timestamp_ns))
    corners_city = ego_pose.apply(corners_ego)
    in_front = frames.is_in_front_of_camera(corners_city, world_to_cam)
    if in_front.sum() < 8:
        return None
    px = frames.world_to_pixel(corners_city, world_to_cam, cam.intr)
    x0, y0 = px[:, 0].min(), px[:, 1].min()
    x1, y1 = px[:, 0].max(), px[:, 1].max()
    W, H = cam.intr.width, cam.intr.height
    x0, x1 = np.clip([x0, x1], 0, W - 1)
    y0, y1 = np.clip([y0, y1], 0, H - 1)
    if (x1 - x0) < 4 or (y1 - y0) < 4:
        return None
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/av2_masks"))
    ap.add_argument("--cameras", nargs="*", default=list(RING_CAMERAS))
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/sam2.1_hiera_small.pt"))
    ap.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    ap.add_argument("--only-moving", action="store_true", default=True)
    ap.add_argument("--all-objects", dest="only_moving", action="store_false")
    ap.add_argument("--min-disp-m", type=float, default=1.5)
    ap.add_argument("--dilate-px", type=int, default=9, help="grow masks to catch shadows/edges")
    args = ap.parse_args()

    seq = Av2Sequence(args.seq_dir)
    assert_av2_frame_sane(seq)

    moving = seq.moving_track_ids(args.min_disp_m) if args.only_moving else None
    if moving is not None:
        n_tracks = seq.annotations.track_uuid.nunique()
        print(f"masking {len(moving)}/{n_tracks} tracks that actually move "
              f"(>{args.min_disp_m}m in the city frame); parked vehicles are kept "
              "as static background")

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(args.model_cfg, str(args.checkpoint), device=device)
    predictor = SAM2ImagePredictor(model)
    print(f"SAM2 loaded on {device}")

    total_imgs = 0
    total_boxes = 0
    for cam_name in args.cameras:
        cam = seq.camera(cam_name)
        out_cam = args.out_dir / cam_name
        out_cam.mkdir(parents=True, exist_ok=True)
        timestamps = seq.image_timestamps(cam_name)
        print(f"{cam_name}: {len(timestamps)} frames")

        for n, ts in enumerate(timestamps):
            img_path = seq.image_path(cam_name, ts)
            img = cv2.imread(str(img_path))
            if img is None:
                raise FileNotFoundError(f"cv2 could not read {img_path}")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H, W = img.shape[:2]

            world_to_cam = seq.world_to_cam(cam, ts)
            rows = seq.cuboids_at(ts)
            if moving is not None:
                rows = rows[rows.track_uuid.isin(moving)]
            boxes = []
            for _, row in rows.iterrows():
                b = cuboid_to_box_prompt(seq, cam, row, world_to_cam)
                if b is not None:
                    boxes.append(b)

            mask = np.zeros((H, W), dtype=np.uint8)
            if boxes:
                boxes_arr = np.stack(boxes)
                predictor.set_image(img_rgb)
                with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
                    m, _, _ = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=boxes_arr,
                        multimask_output=False,
                    )
                m = np.asarray(m)
                if m.ndim == 4:
                    m = m[:, 0]
                elif m.ndim == 2:
                    m = m[None]
                mask = (m.any(axis=0) * 255).astype(np.uint8)
                total_boxes += len(boxes)

            if args.dilate_px > 0:
                k = np.ones((args.dilate_px, args.dilate_px), np.uint8)
                mask = cv2.dilate(mask, k, iterations=1)

            cv2.imwrite(str(out_cam / f"{ts}.png"), mask)
            total_imgs += 1
            if n % 50 == 0:
                cov = (mask > 0).mean()
                print(f"  [{n}/{len(timestamps)}] boxes={len(boxes)} masked={cov:.1%}")

    print(f"done: {total_imgs} masks, {total_boxes} cuboid prompts -> {args.out_dir}")


if __name__ == "__main__":
    main()
