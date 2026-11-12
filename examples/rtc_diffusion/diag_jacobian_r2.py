#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Diagnose PiGDM's r2 assumption on a trained DiffusionPolicy: is J / r2 = sqrt(a) / (1 - a)?

For each sampled dataset frame (observation history ending at that frame):
  1. K unguided chunks are sampled from the same observation -> per-element conditional variance s2 and
     full covariance Sigma (normalized action space, raw horizon x action dims, D = horizon * A).
  2. Along one unguided DDIM trajectory, at every denoising step the full Jacobian J = d x0_hat / d x_t
     (D x D) of the trained denoiser is computed, and compared with
        * unit prior (paper):        J_ii / (1 - a)        vs  G = sqrt(a) / (1 - a)
        * Gaussian posterior r2:     J_ii / r2_i, r2_i = s2_i (1 - a) / (a s2_i + 1 - a)   vs  G
        * full Gaussian prior:       J  vs  sqrt(a) Sigma (a Sigma + (1 - a) I)^-1
        * symmetry: the exact Tweedie Jacobian sqrt(a) / (1 - a) Cov[x0 | x_t] is symmetric (and PSD).
        * conditioning (epsilon prediction): sqrt(1 - a) d eps / d x_t = I - sqrt(a) J, so J is the residual
          of a near-identity map; `sqrt(a) |J|_F / sqrt(D)` is that residual for the model and for the
          Gaussian J. If the true residual is far below the network's relative accuracy, J is noise.
  3. (optional) M stochastic DDPM (ancestral) completions from the same x_t give the model's own posterior
     Cov_anc[x0 | x_t]. Tweedie's second-order identity says Cov[x0 | x_t] = (1 - a) / sqrt(a) * J exactly
     (any prior), so C_J = (1 - a) / sqrt(a) * J_sym vs Cov_anc tests whether the learned Jacobian is
     consistent with the model's own samples, and r2_i vs Cov_anc_ii tests the Gaussian-s2 approximation.
     The last DDPM step (t = 0) is deterministic, so this check is skipped at t = 0.

Ratios are reported as median [p10, p90] over elements and frames, per denoising step. Ideal value: 1.

Example (PushT):
    python examples/rtc_diffusion/diag_jacobian_r2.py --policy.path=<migrated diffusion_pusht dir> \
        --policy.noise_scheduler_type=DDIM --policy.num_inference_steps=10 \
        --dataset.repo_id=lerobot/pusht --num_frames=24 --num_var_samples=64 --num_posterior_samples=64
"""

import csv
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from lerobot.configs import DatasetConfig, PreTrainedConfig, parser
from lerobot.datasets import LeRobotDataset
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.diffusion.modeling_diffusion import _make_noise_scheduler
from lerobot.policies.diffusion.rtc_diffusion import (
    posterior_r2,
    push_observation_to_queues,
    snapshot_observations,
    tweedie_x0,
)
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging


@dataclass
class JacobianR2DiagConfig:
    policy: PreTrainedConfig | None = None
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    num_frames: int = 24
    # Chunks per observation used to estimate s2 / Sigma.
    num_var_samples: int = 64
    # Ancestral DDPM completions per (frame, step) for Cov_anc[x0 | x_t]; 0 disables the check.
    num_posterior_samples: int = 64
    device: str | None = None
    seed: int = 7
    output_dir: str = "outputs/rtc_diffusion_r2_diag"

    def __post_init__(self) -> None:
        policy_path = parser.get_path_arg("policy")
        if not policy_path:
            raise ValueError("--policy.path is required")
        self.policy = PreTrainedConfig.from_pretrained(
            policy_path, cli_overrides=parser.get_cli_overrides("policy") or []
        )
        self.policy.pretrained_path = policy_path
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


def _batched(item: dict) -> dict:
    return {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v]) for k, v in item.items()}


def full_jacobian(
    model, x_t: torch.Tensor, t: int, global_cond: torch.Tensor, alpha_bar: float
) -> torch.Tensor:
    """J[i, j] = d x0_hat_i / d x_t_j (D x D) with one batched backward: D independent copies of x_t, copy i
    gets the cotangent e_i (GroupNorm / eval mode: batch elements do not interact)."""
    d = x_t[0].numel()
    with torch.enable_grad():
        x = x_t.detach().expand(d, *x_t.shape[1:]).clone().requires_grad_(True)
        timestep = torch.full((d,), t, dtype=torch.long, device=x.device)
        out = model.unet(x, timestep, global_cond=global_cond.expand(d, -1))
        x0 = tweedie_x0(out, x, alpha_bar, model.config.prediction_type)
        eye = torch.eye(d, device=x.device, dtype=x.dtype).view(d, *x_t.shape[1:])
        (rows,) = torch.autograd.grad(x0, x, grad_outputs=eye)
    return rows.reshape(d, d).double()


@torch.no_grad()
def ancestral_completions(model, ddpm, x_t: torch.Tensor, t: int, global_cond, m: int, gen) -> torch.Tensor:
    """M samples of x0 ~ p_model(x0 | x_t) with the DDPM reverse chain t, t-1, ..., 0 (stride 1)."""
    x = x_t.expand(m, *x_t.shape[1:]).clone()
    cond = global_cond.expand(m, -1)
    for tau in range(t, -1, -1):
        timestep = torch.full((m,), tau, dtype=torch.long, device=x.device)
        out = model.unet(x, timestep, global_cond=cond)
        x = ddpm.step(out, tau, x, generator=gen).prev_sample
    return x


def _q(x: np.ndarray) -> tuple[float, float, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (float("nan"),) * 3
    return float(np.median(x)), float(np.percentile(x, 10)), float(np.percentile(x, 90))


def main_loop(cfg: JacobianR2DiagConfig) -> None:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset(cfg.dataset.repo_id)
    policy = get_policy_class(cfg.policy.type).from_pretrained(cfg.policy.pretrained_path, config=cfg.policy)
    policy = policy.to(cfg.device).eval()
    pre, _ = make_pre_post_processors(
        cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": cfg.device}},
    )
    model, conf = policy.diffusion, policy.config
    sched = model.noise_scheduler
    ddpm = _make_noise_scheduler(
        "DDPM",
        num_train_timesteps=conf.num_train_timesteps,
        beta_start=conf.beta_start,
        beta_end=conf.beta_end,
        beta_schedule=conf.beta_schedule,
        clip_sample=conf.clip_sample,
        clip_sample_range=conf.clip_sample_range,
        prediction_type=conf.prediction_type,
    )
    ddpm.set_timesteps(conf.num_train_timesteps)
    horizon, action_dim = conf.horizon, conf.action_feature.shape[0]
    dim = horizon * action_dim
    logging.info(
        "policy %s | %s %d steps | horizon=%d A=%d D=%d | clip_sample=%s",
        cfg.policy.pretrained_path,
        conf.noise_scheduler_type,
        model.num_inference_steps,
        horizon,
        action_dim,
        dim,
        conf.clip_sample,
    )

    ep = np.asarray(dataset.hf_dataset["episode_index"])
    starts = {int(ep[s]): int(s) for s in np.r_[0, np.flatnonzero(np.diff(ep)) + 1]}
    frames = sorted(random.Random(cfg.seed).sample(range(len(ep)), cfg.num_frames))

    def obs(j: int) -> dict:
        item = {k: v for k, v in dataset[j].items() if k.startswith("observation.") or k == "task"}
        return pre(_batched(item))

    rows = []
    for fi, k in enumerate(frames):
        policy.reset()
        for j in range(max(k - conf.n_obs_steps + 1, starts[int(ep[k])]), k + 1):
            push_observation_to_queues(policy, obs(j))
        with torch.no_grad():
            global_cond = model._prepare_global_conditioning(snapshot_observations(policy))
            gen = torch.Generator(device=cfg.device).manual_seed(cfg.seed * 1000 + fi)
            samples = model.conditional_sample(
                cfg.num_var_samples, global_cond=global_cond.expand(cfg.num_var_samples, -1), generator=gen
            )
            flat = samples.reshape(cfg.num_var_samples, dim).double()
            sigma = torch.cov(flat.T)  # (D, D), unbiased
            s2 = sigma.diagonal().clamp_min(1e-12)
            x_t = torch.randn(1, horizon, action_dim, device=cfg.device, generator=gen)

        sched.set_timesteps(model.num_inference_steps)
        eye = torch.eye(dim, dtype=torch.float64, device=cfg.device)
        for step, t in enumerate(sched.timesteps):
            t = int(t)
            a = float(sched.alphas_cumprod[t])
            g_ideal = a**0.5 / (1 - a)
            jac = full_jacobian(model, x_t, t, global_cond, a)
            j_sym = 0.5 * (jac + jac.T)
            jd = jac.diagonal()
            r2_post = posterior_r2(a, s2)
            # Gaussian prior N(mu, Sigma): J = sqrt(a) Sigma (a Sigma + (1 - a) I)^-1
            j_gauss = a**0.5 * torch.linalg.solve(a * sigma + (1 - a) * eye, sigma.T).T
            row = {
                "frame": k,
                "step": step,
                "timestep": t,
                "alpha_bar": a,
                "G": g_ideal,
                "s2_median": float(s2.median()),
                "s2_min": float(s2.min()),
                "noise_floor_ratio": (1 - a) / a / float(s2.median()),  # >> 1: unit vs posterior r2 differ
            }
            for name, vals in {
                "ratio_unit": (jd / (1 - a)).cpu().numpy() / g_ideal,
                "ratio_post": (jd / r2_post).cpu().numpy() / g_ideal,
                "ratio_gauss_full_diag": (jd / j_gauss.diagonal()).cpu().numpy(),
            }.items():
                row[name], row[name + "_p10"], row[name + "_p90"] = _q(vals)
            row["relerr_gauss_full"] = float((j_sym - j_gauss).norm() / j_gauss.norm())
            row["asym"] = float((jac - jac.T).norm() / (jac + jac.T).norm())
            row["top_eig_ratio"] = float(
                torch.linalg.eigvalsh(j_sym)[-1] / torch.linalg.eigvalsh(j_gauss)[-1]
            )
            row["trace_ratio_gauss_full"] = float(jac.trace() / j_gauss.trace())
            # epsilon parameterization: sqrt(1-a) d eps / d x_t = I - sqrt(a) J. J is the small residual of a
            # near-identity map; these are |I - sqrt(1-a) d eps/dx|_F / sqrt(D) for the model and for Gaussian J.
            row["resid_model"] = a**0.5 * float(jac.norm()) / dim**0.5
            row["resid_gauss"] = a**0.5 * float(j_gauss.norm()) / dim**0.5

            if cfg.num_posterior_samples > 1 and t > 0:
                pgen = torch.Generator(device=cfg.device).manual_seed(cfg.seed * 100000 + fi * 100 + step)
                comp = ancestral_completions(
                    model, ddpm, x_t, t, global_cond, cfg.num_posterior_samples, pgen
                )
                comp = comp.reshape(cfg.num_posterior_samples, dim).double()
                cov_anc = torch.cov(comp.T)
                c_j = (1 - a) / a**0.5 * j_sym
                with torch.no_grad():
                    out = model.unet(
                        x_t, torch.full((1,), t, dtype=torch.long, device=x_t.device), global_cond=global_cond
                    )
                    x0_hat = tweedie_x0(out, x_t, a, conf.prediction_type).reshape(dim).double()
                ad = cov_anc.diagonal().clamp_min(1e-12)
                for name, vals in {
                    "cj_over_anc": (c_j.diagonal() / ad).cpu().numpy(),
                    "r2post_over_anc": (r2_post / ad).cpu().numpy(),
                    "r2unit_over_anc": ((1 - a) / ad).cpu().numpy(),
                }.items():
                    row[name], row[name + "_p10"], row[name + "_p90"] = _q(vals)
                row["trace_cj_over_anc"] = float(c_j.trace() / cov_anc.trace())
                row["anc_sd_rms"] = float(ad.mean().sqrt())
                row["anc_mean_minus_x0hat_rms"] = float((comp.mean(0) - x0_hat).pow(2).mean().sqrt())
            rows.append(row)

            with torch.no_grad():  # advance the unguided DDIM trajectory
                out = model.unet(
                    x_t, torch.full((1,), t, dtype=torch.long, device=x_t.device), global_cond=global_cond
                )
                x_t = sched.step(out, t, x_t).prev_sample
        logging.info(
            "[%d/%d] frame %d s2 median %.2e  ratio_post(step0..9) %s",
            fi + 1,
            len(frames),
            k,
            float(s2.median()),
            " ".join(f"{r['ratio_post']:.2f}" for r in rows[-len(sched.timesteps) :]),
        )

    keys = sorted({kk for r in rows for kk in r}, key=lambda x: (x not in ("frame", "step", "timestep"), x))
    with open(out_dir / "jacobian_r2_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # Per-step summary: medians over frames of the per-frame medians (p10/p90 over frames of the medians).
    cols = [
        ("alpha_bar", "a"),
        ("noise_floor_ratio", "(1-a)/(a s2)"),
        ("ratio_unit", "J/(1-a) / G"),
        ("ratio_post", "J/r2_post / G"),
        ("ratio_gauss_full_diag", "J_ii/Jgauss_ii"),
        ("relerr_gauss_full", "|J-Jgauss|/|Jgauss|"),
        ("asym", "asym"),
        ("resid_model", "sqrt(a)|J|/sqrt(D)"),
        ("resid_gauss", "same, Gaussian"),
        ("cj_over_anc", "C_J/Cov_anc"),
        ("r2post_over_anc", "r2_post/Cov_anc"),
        ("r2unit_over_anc", "(1-a)/Cov_anc"),
        ("anc_sd_rms", "sd_anc"),
    ]
    lines = [
        "| step | t | " + " | ".join(c[1] for c in cols) + " |",
        "|---|---|" + "---|" * len(cols),
    ]
    steps = sorted({r["step"] for r in rows})
    for st in steps:
        sel = [r for r in rows if r["step"] == st]
        cells = []
        for key, _ in cols:
            vals = np.array([r.get(key, float("nan")) for r in sel], dtype=float)
            med, lo, hi = _q(vals)
            cells.append("-" if np.isnan(med) else f"{med:.3g} [{lo:.2g}, {hi:.2g}]")
        lines.append(f"| {st} | {sel[0]['timestep']} | " + " | ".join(cells) + " |")
    s2_all = np.array([r["s2_median"] for r in rows if r["step"] == 0])
    header = (
        f"frames={len(frames)} K={cfg.num_var_samples} M={cfg.num_posterior_samples} | "
        f"per-frame median s2: median {np.median(s2_all):.3g} [p10 {np.percentile(s2_all, 10):.3g}, "
        f"p90 {np.percentile(s2_all, 90):.3g}]\n"
        "cells: median over frames of the per-frame median over elements [p10, p90 over frames]\n\n"
    )
    table = header + "\n".join(lines) + "\n"
    (out_dir / "jacobian_r2_summary.md").write_text(table)
    print(table)


@parser.wrap()
def main(cfg: JacobianR2DiagConfig) -> None:
    init_logging()
    set_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main_loop(cfg)


if __name__ == "__main__":
    main()
