"""GPU box: build a reasoner-only local copy of nvidia/Cosmos3-Nano that fits the disk.

Every transformer shard interleaves understanding weights with the generator's MoT expert
(moe_gen, add_[qkv]_proj, ...), so a plain download is 31.5 GB. Cosmos3OmniForConditionalGeneration
never instantiates the generator -- it lists those keys in _keys_to_ignore_on_load_unexpected
and drops them on load. This script applies THAT SAME list (read from the installed class, not
hard-coded) shard by shard: download one shard, keep the tensors the class loads under their
original names, delete the original, next. Peak disk ~ kept-so-far + one shard.

Output dir has the repo's layout, so from_pretrained(out_dir) takes the same loading path as
from_pretrained("nvidia/Cosmos3-Nano"). The final check loads it and requires ZERO missing keys.

  python trim_cosmos3.py --out /workspace/cosmos3_nano_und
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Cosmos3OmniForConditionalGeneration

REPO = "nvidia/Cosmos3-Nano"
p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--tmp", default="/workspace/_dl_tmp")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

IGNORE = [re.compile(x) for x in Cosmos3OmniForConditionalGeneration._keys_to_ignore_on_load_unexpected]
print("ignore patterns (from installed class):", [x.pattern for x in IGNORE])
dropped = lambda k: any(r.search(k) for r in IGNORE)

files = [f.rfilename for f in HfApi().model_info(REPO, files_metadata=False).siblings]
SKIP_DIRS = ("assets/", "vae/", "sound_tokenizer/", "scheduler/")
small = [f for f in files if not f.endswith(".safetensors") and not f.startswith(SKIP_DIRS)
         and not f.endswith((".md", ".mp4", ".png", ".jpg"))]
for f in small:
    hf_hub_download(REPO, f, local_dir=a.out)
print(f"fetched {len(small)} config/tokenizer/processor files")

idx = json.load(open(os.path.join(a.out, "model.safetensors.index.json")))
wm = idx["weight_map"]
keep_map, total = {}, 0
for shard in sorted(set(wm.values())):
    src = hf_hub_download(REPO, shard, local_dir=a.tmp)
    kept = {}
    with safe_open(src, "pt") as fh:
        meta = fh.metadata()
        for k in fh.keys():
            if not dropped(k):
                kept[k] = fh.get_tensor(k)
    dst = os.path.join(a.out, shard)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if kept:
        save_file(kept, dst, metadata=meta or {"format": "pt"})
        for k, t in kept.items():
            keep_map[k] = shard
            total += t.numel() * t.element_size()
    os.remove(src)
    shutil.rmtree(os.path.join(a.tmp, ".cache"), ignore_errors=True)
    print(f"  {shard}: kept {len(kept)} tensors | disk free "
          f"{shutil.disk_usage('/').free / 1e9:.1f} GB", flush=True)
    del kept

idx["weight_map"] = keep_map
idx.setdefault("metadata", {})["total_size"] = total
json.dump(idx, open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=2)
shutil.rmtree(a.tmp, ignore_errors=True)
print(f"trimmed checkpoint: {len(keep_map)} tensors, {total / 1e9:.2f} GB")

# ---- proof: same class, same loader, nothing missing ----
m, info = Cosmos3OmniForConditionalGeneration.from_pretrained(
    a.out, dtype=torch.bfloat16, low_cpu_mem_usage=True, output_loading_info=True)
print("missing_keys:", info["missing_keys"])
print("unexpected_keys:", sorted(info["unexpected_keys"])[:5], len(info["unexpected_keys"]))
print("mismatched_keys:", info["mismatched_keys"])
assert not info["missing_keys"], "trim removed weights the model needs -- DO NOT USE"
assert not info["mismatched_keys"]
print(f"LOAD OK: {sum(p.numel() for p in m.parameters()) / 1e9:.2f} B params")
