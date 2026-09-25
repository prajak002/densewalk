# DenseWalk — Full Pipeline

## Goal
Dense crowd video → 4D Gaussian scene reconstruction → 360 depth
perception → Unitree G1 navigating through the splat environment
avoiding collisions.

## Pipeline order — do not skip steps
1. Setup
2. 3DGS static background
3. 4D crowd reconstruction
4. 360 panoramic render
5. G1 locomotion
6. G1 navigation in splat scene

## Hard rules
- Check disk before every step: df -h / — stop if under 20GB free
- Copy every artifact to Mac immediately after it is created
- Stop after each artifact and show me before continuing
- Splats are VISUAL ONLY — never physics colliders
- G1 needs ground plane collider or falls through world
- Never guess APIs — read installed package first
- All coordinate conversions through frames.py
- Metres, Z-up, floor at z=0

## Machine
SSH: ssh -p 51081 root@93.91.156.106
GPU: RTX PRO 6000 Blackwell Max-Q, 96GB VRAM, 384 cores, 76GB disk
Do NOT destroy the instance without explicit confirmation.

## Absolute rules
1. scp every artifact to Mac before next milestone
2. Check disk before every milestone
3. Show nohup script before running any long job
4. Never guess Isaac Sim / gsplat / MoSca APIs — read installed package first
5. If blocked, say exactly what broke and stop
6. Never proceed past a broken step silently
7. One milestone at a time
8. Wait for go-ahead after each artifact

## State as of 2026-09-18
- AV2 sequence + SAM2 masks were lost when earlier instances were reclaimed;
  AV2 is re-downloadable from its public S3 bucket, masks are regenerable.
- G1 policy (1500 iters, mean reward 28.09) rescued to the Mac at
  outputs/g1_policy/{policy.onnx,model_1499.pt} — reuse at M4, do not retrain.
- Sequence in use: 04994d08-156c-3018-9717-ba0e29be8153
