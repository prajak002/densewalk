"""GPU box ONLY: one backbone pass per keyframe -> latents.npz for the horizon head.

Model/processor/encode/command_latent calls are copied from the DCA notebook's v2 code
(Cosmos3_Nano_DCA_stable_v3_vastai.ipynb cells 10 + 14), which produced the V2_full weights.
Nothing here is a guessed API -- but run `--limit 5` first and eyeball the output.

Saves, per image path (same key as the notebook's Z0_CACHE):
    z0   LoRA OFF last-token hidden     (ablation)
    za   LoRA ON  last-token hidden     (ablation)
    h    hcmd actually fed to the DCA heads:
           dca_full / *_prior rungs: z0 + sigmoid(gate(z0)) * (za - z0)
           lora_only / lora_unsafe : za
    dca_v, dca_dir -- the checkpoint's own current-state outputs, for baseline B1b and to
                     cross-check against predictions_V2_full_<rung>.json

  python scripts/cache_dca_latents.py \
      --run-dir /workspace/V2_full --rung dca_full --ckpt best \
      --json '/workspace/dw_data/json_openvla*/*.json' --data-root /workspace/dw_data \
      --out /workspace/latents_dca_full.npz
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--run-dir", required=True, help=".../V2_full")
p.add_argument("--rung", default="dca_full")
p.add_argument("--ckpt", default="best", choices=["best", "final"])
p.add_argument("--json", required=True)
p.add_argument("--data-root", required=True)
p.add_argument("--base-model", default="nvidia/Cosmos3-Nano")
p.add_argument("--vel-reps", default=None, help="comma list; default = read from metrics json if present")
p.add_argument("--out", required=True)
p.add_argument("--limit", type=int, default=0)
a = p.parse_args()

DEV = "cuda:0"
GATED = a.rung in ("dca_full", "lora_unsafe_prior")

from peft import PeftModel                                                   # noqa: E402
from transformers import AutoProcessor, Cosmos3OmniForConditionalGeneration  # noqa: E402

processor = AutoProcessor.from_pretrained(a.base_model)
base = Cosmos3OmniForConditionalGeneration.from_pretrained(
    a.base_model, dtype=torch.bfloat16, low_cpu_mem_usage=True).to(DEV)
adir = os.path.join(a.run_dir, a.rung, f"adapter_{a.ckpt}")
vla = PeftModel.from_pretrained(base, adir).eval()
heads_sd = torch.load(os.path.join(a.run_dir, a.rung, f"heads_{a.ckpt}.pt"), map_location=DEV)
print("heads keys:", sorted(heads_sd))


def lin(name):
    return heads_sd[f"{name}.weight"].float(), heads_sd[f"{name}.bias"].float()


def encode(image_path, instr):                     # notebook cell 10, verbatim logic
    messages = [{"role": "user",
                 "content": [{"type": "image", "path": str(image_path)},
                             {"type": "text", "text": instr.strip()}]}]
    enc = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                        return_dict=True, return_tensors="pt")
    return {k: (v.to(DEV, dtype=torch.bfloat16) if k == "pixel_values" else v.to(DEV))
            for k, v in enc.items()}


def command_latent(item, use_base):                # notebook cell 10, verbatim logic
    ctx = vla.disable_adapter() if use_base else contextlib.nullcontext()
    kw = dict(input_ids=item["input_ids"], attention_mask=item["attention_mask"],
              pixel_values=item["pixel_values"], output_hidden_states=True, use_cache=False)
    for k in ("image_grid_thw", "mm_token_type_ids"):
        if k in item:
            kw[k] = item[k]
    with ctx:
        out = vla(**kw)
    return out.hidden_states[-1][:, -1, :].float()


def resolve_img(pth):                               # notebook cell 8 resolve_img
    b, vid = os.path.basename(pth), os.path.basename(os.path.dirname(pth))
    cands = [pth, os.path.join(a.data_root, pth)]
    cands += [os.path.join(d, vid, b) for d in glob.glob(os.path.join(a.data_root, "keyframes*"))]
    for c in cands:
        if os.path.exists(c):
            return c
    return None


vel_reps = None
if a.vel_reps:
    vel_reps = [float(x) for x in a.vel_reps.split(",")]

keys, Z0, ZA, H, V, D, OD, G = [], [], [], [], [], [], [], []
todo = []
for jp in sorted(glob.glob(a.json)):
    d = json.load(open(jp))
    instr = d["instruction"]["text"] if isinstance(d["instruction"], dict) else d["instruction"]
    for f in d["frames"]:
        todo.append((f["image"], instr))
if a.limit:
    todo = todo[: a.limit]
print(f"{len(todo)} frames to encode", flush=True)
t0, missing = time.time(), 0
with torch.no_grad():
    for n, (key, instr) in enumerate(todo):
        img = resolve_img(key)
        if img is None:
            missing += 1
            continue
        it = encode(img, instr)
        za = command_latent(it, use_base=False)
        z0 = command_latent(it, use_base=True)
        gw, gb = lin("gate")
        if GATED:                                  # notebook commands(): gate reads DETACHED z0
            g = torch.sigmoid(F.linear(z0, gw, gb))
            hc = z0 + g * (za - z0)
        else:                                      # notebook: gate = sigmoid(gate(adapted)); hcmd = adapted
            g = torch.sigmoid(F.linear(za, gw, gb))
            hc = za
        # affordance heads read z_aff: base z0 when afford_decouple, else adapted/hcmd.
        # V2_full predates afford_decouple and ran with afford_raw=True -> z_aff = adapted
        ow, ob = lin("obs_dist")
        od = F.linear(za, ow, ob) * 10.0            # CFG["obs_norm_m"] = 10
        dw, db = lin("dir")
        dv = F.linear(hc, dw, db); dv = dv / (dv.norm(dim=-1, keepdim=True) + 1e-6)
        if vel_reps is not None and "vel_ord.weight" in heads_sd:
            vw, vb = lin("vel_ord")
            v = (F.softmax(F.linear(hc, vw, vb), -1) * torch.tensor(vel_reps, device=DEV)).sum(-1)
        else:
            v = torch.full((1,), float("nan"), device=DEV)
        keys.append(key)
        Z0.append(z0[0].half().cpu().numpy()); ZA.append(za[0].half().cpu().numpy())
        H.append(hc[0].half().cpu().numpy())
        V.append(float(v[0])); D.append(dv[0].cpu().numpy())
        OD.append(float(od[0, 0])); G.append(float(g[0, 0]))
        if (n + 1) % 250 == 0:
            print(f"  {n + 1}/{len(todo)}  {(time.time() - t0) / 60:.1f} min", flush=True)
np.savez(a.out, images=np.array(keys), h=np.stack(H), z0=np.stack(Z0), za=np.stack(ZA),
         dca_v=np.array(V, dtype=np.float32), dca_dir=np.stack(D).astype(np.float32),
         dca_obs_dist_m=np.array(OD, dtype=np.float32), dca_gate=np.array(G, dtype=np.float32))
print(f"saved {a.out}: {len(keys)} frames, {missing} missing images, {(time.time() - t0) / 60:.1f} min")
