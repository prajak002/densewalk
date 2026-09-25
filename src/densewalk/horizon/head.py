"""Δt-conditioned horizon head over FROZEN DCA latents.

    d̂(t+Δt) = CV(s, Δt) + Δt · r_xy(h, s, Δt)       Laplace scale  b_xy = Δt·softplus(·) + ε
    ĉ(t+Δt) = softplus(clr_t + r_c(h, s, Δt))         Laplace scale  b_c  = softplus(·) + ε

Δt enters ONLY through a log-time Fourier embedding that produces FiLM (γ, β) for each
residual block. The FiLM generator and the output layer are zero-initialised, so at step 0
the head IS the constant-velocity predictor -- training can only learn deviations from it.

Variants (same class, so every baseline shares code and capacity):
    cond="film"   recommended
    cond="concat" time embedding concatenated to the input (ablation F)
    cond="none"   no Δt input; train one per horizon (baseline B3)
    use_h=False   state-only, no image latent (baseline B2 -- the critical control)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# s = [v, cos a, sin a, clr, clr_valid]; clearance rescaled so inputs are O(1)
S_SCALE = (1.0, 1.0, 1.0, 0.1, 1.0)
EPS = 1e-2
LAPLACE_K68 = math.log(1 / 0.32)   # |e| < k·b holds with prob 1 - e^{-k}
LAPLACE_K95 = math.log(20.0)


class LogTimeEmb(nn.Module):
    """Horizons span ~1.5 decades, so embed log Δt, not Δt."""

    def __init__(self, n_freq: int = 4, t_min: float = 0.05, t_max: float = 4.0):
        super().__init__()
        self.lo, self.hi = math.log(t_min), math.log(t_max)
        self.register_buffer("f", math.pi * 2.0 ** torch.arange(n_freq, dtype=torch.float32))
        self.dim = 1 + 2 * n_freq

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        u = ((dt.clamp_min(1e-3).log() - self.lo) / (self.hi - self.lo))[:, None]
        return torch.cat([u, (u * self.f).sin(), (u * self.f).cos()], -1)


class FiLMBlock(nn.Module):
    def __init__(self, d: int, p_drop: float):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.SiLU(), nn.Dropout(p_drop), nn.Linear(2 * d, d))

    def forward(self, x, g, b):
        return x + self.mlp((1 + g) * self.ln(x) + b)


class HorizonHead(nn.Module):
    def __init__(self, d_h: int = 4096, d_s: int = 5, d: int = 256, n_blocks: int = 2,
                 cond: str = "film", use_h: bool = True, p_drop: float = 0.1):
        super().__init__()
        assert cond in ("film", "concat", "none")
        self.cond, self.use_h, self.n_blocks = cond, use_h, n_blocks
        self.te = LogTimeEmb()
        self.register_buffer("s_scale", torch.tensor(S_SCALE[:d_s]))
        if use_h:
            self.h_in = nn.Sequential(nn.LayerNorm(d_h), nn.Dropout(p_drop), nn.Linear(d_h, d))
        self.s_in = nn.Linear(d_s + (self.te.dim if cond == "concat" else 0), d)
        if cond == "film":
            self.film = nn.Sequential(nn.Linear(self.te.dim, d), nn.SiLU(), nn.Linear(d, 2 * d * n_blocks))
            nn.init.zeros_(self.film[-1].weight); nn.init.zeros_(self.film[-1].bias)
        self.blocks = nn.ModuleList(FiLMBlock(d, p_drop) for _ in range(n_blocks))
        self.out = nn.Linear(d, 6)          # r_x, r_y, r_c, raw_bx, raw_by, raw_bc
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, h: torch.Tensor | None, s: torch.Tensor, dt: torch.Tensor) -> dict:
        sn = s * self.s_scale
        if self.cond == "concat":
            sn = torch.cat([sn, self.te(dt)], -1)
        x = self.s_in(sn)
        if self.use_h:
            x = x + self.h_in(h.float())
        if self.cond == "film":
            gb = self.film(self.te(dt)).chunk(2 * self.n_blocks, -1)
        for k, blk in enumerate(self.blocks):
            if self.cond == "film":
                x = blk(x, gb[2 * k], gb[2 * k + 1])
            else:
                x = blk(x, 0.0, 0.0)
        o, t = self.out(x), dt[:, None]
        cv = cv_displacement(s, dt)
        clr0 = s[:, 3]
        return {
            "disp": cv + t * o[:, :2],                                   # exact 0 at Δt = 0
            "b_xy": t * F.softplus(o[:, 3:5]) + EPS,                    # 0.69·Δt at init
            "clr": F.softplus(clr0 + o[:, 2]),
            "b_c": F.softplus(o[:, 5]) + EPS,
        }


def cv_displacement(s: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """Constant-velocity extrapolation from the ego state (baseline B1)."""
    return s[:, :1] * dt[:, None] * s[:, 1:3]


def horizon_loss(p: dict, y: dict, lam_c: float = 1.0, beta: float = 0.0):
    """Laplace NLL on displacement + masked Laplace NLL on clearance.
    beta>0 -> β-NLL (per-sample weight b^β, stop-grad) if the scales start inflating."""
    e = (y["disp"] - p["disp"]).abs()
    nll_d = e / p["b_xy"] + p["b_xy"].log()
    if beta:
        nll_d = nll_d * p["b_xy"].detach() ** beta
    l_d = nll_d.sum(-1).mean()
    m = y["clr_valid_t1"]
    nll_c = (y["clr_t1"] - p["clr"]).abs() / p["b_c"] + p["b_c"].log()
    if beta:
        nll_c = nll_c * p["b_c"].detach() ** beta
    l_c = (m * nll_c).sum() / m.sum().clamp_min(1.0)
    return l_d + lam_c * l_c, {"disp": float(l_d.detach()), "clr": float(l_c.detach())}


@torch.no_grad()
def predict_horizons(head: HorizonHead, h: torch.Tensor, s: torch.Tensor,
                     horizons=(0.1, 0.5, 1.0, 2.0, 3.0)) -> dict:
    """One frame, any set of Δt, one batched call. h: [1, d_h], s: [1, 5]."""
    dt = torch.tensor(horizons, dtype=torch.float32, device=s.device)
    n = len(horizons)
    return head(None if h is None else h.expand(n, -1), s.expand(n, -1), dt)
