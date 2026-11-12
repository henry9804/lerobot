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

"""Offline Real-Time Chunking check for DiffusionPolicy on dataset frames.

For each sampled frame k of an episode:
  1. chunk A is generated from the observation history ending at frame k (no RTC);
  2. `replan_every` (s) steps later, chunk B is generated from the history ending at frame k + s with
     `prev_chunk_left_over = A[s:]` (normalized), once per variant (naive / guidance / replacement ...),
     always from the same initial noise;
  3. actions are plotted per dimension (env units) together with the ground-truth dataset actions, and
     metrics are written to CSV: RMS mismatch to the leftover on the frozen prefix [0, d) and on the
     whole overlap, the step at the switch ||B[d] - A[s + d - 1]||, max second difference inside B,
     error to the dataset actions, and inference latency. Per-denoising-step guidance statistics
     (alpha_bar, w, ||e||, ||g||, ||delta||) are saved for the guided variants.

Example (PushT):
    python examples/rtc_diffusion/eval_dataset.py \
        --policy.path=<migrated diffusion_pusht dir> \
        --policy.noise_scheduler_type=DDIM --policy.num_inference_steps=10 \
        --dataset.repo_id=lerobot/pusht --inference_delay=3 --replan_every=8 \
        --rtc.max_guidance_weight=10 --num_samples=16 --output_dir=outputs/rtc_diffusion_offline
"""

import csv
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from lerobot.configs import DatasetConfig, PreTrainedConfig, RTCAttentionSchedule, parser
from lerobot.datasets import LeRobotDataset
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc import RTCConfig
from lerobot.utils.constants import ACTION
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

# variant -> (rtc_guidance_mode, schedule, jacobian, weight clip or None = cfg.weight_clip, r2, r2 clip)
# "*_post*": r2 = per-element Gaussian posterior variance from `r2_num_samples` unguided chunks
# (rtc_guidance_r2="posterior"); "_lit" = beta clips 1 / r2 itself (rtc_r2_clip="literal").
VARIANTS: dict[str, tuple[str, RTCAttentionSchedule, str, str | None, str, str]] = {
    "naive": ("none", RTCAttentionSchedule.ZEROS, "vjp", None, "unit", "relative"),
    "guidance_zeros": ("guidance", RTCAttentionSchedule.ZEROS, "vjp", None, "unit", "relative"),
    "guidance_exp": ("guidance", RTCAttentionSchedule.EXP, "vjp", None, "unit", "relative"),
    "guidance_zeros_id": ("guidance", RTCAttentionSchedule.ZEROS, "identity", "flow", "unit", "relative"),
    "guidance_exp_id": ("guidance", RTCAttentionSchedule.EXP, "identity", "flow", "unit", "relative"),
    "guidance_exp_post": ("guidance", RTCAttentionSchedule.EXP, "vjp", None, "posterior", "relative"),
    "guidance_exp_post_lit": ("guidance", RTCAttentionSchedule.EXP, "vjp", None, "posterior", "literal"),
    "replacement": ("replacement", RTCAttentionSchedule.ZEROS, "vjp", None, "unit", "relative"),
}
COLORS = {
    "naive": "tab:blue",
    "guidance_zeros": "tab:olive",
    "guidance_exp": "tab:green",
    "guidance_zeros_id": "tab:purple",
    "guidance_exp_id": "tab:pink",
    "guidance_exp_post": "tab:cyan",
    "guidance_exp_post_lit": "tab:brown",
    "replacement": "tab:orange",
}


@dataclass
class OfflineRTCEvalConfig:
    policy: PreTrainedConfig | None = None
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    rtc: RTCConfig = field(
        default_factory=lambda: RTCConfig(
            enabled=True, execution_horizon=100, max_guidance_weight=10.0, debug=True, debug_maxlen=1000
        )
    )
    variants: list[str] = field(
        default_factory=lambda: [v for v in VARIANTS if "_post" not in v]  # posterior r2: opt-in
    )
    weight_clip: str = "pigdm"
    # Chunks per observation for the s2 estimate of the "*_post*" variants.
    r2_num_samples: int = 16
    inference_delay: int = 3
    replan_every: int = 8
    num_samples: int = 8
    num_plots: int = 2
    device: str | None = None
    seed: int = 42
    output_dir: str = "outputs/rtc_diffusion_offline"

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
        unknown = [v for v in self.variants if v not in VARIANTS]
        if unknown:
            raise ValueError(f"Unknown variants {unknown}; choose from {list(VARIANTS)}")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


def _batched(item: dict) -> dict:
    return {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v]) for k, v in item.items()}


class Evaluator:
    def __init__(self, cfg: OfflineRTCEvalConfig):
        self.cfg = cfg
        self.dataset = LeRobotDataset(cfg.dataset.repo_id)
        self.policy = get_policy_class(cfg.policy.type).from_pretrained(
            cfg.policy.pretrained_path, config=cfg.policy
        )
        self.policy = self.policy.to(cfg.device).eval()
        self.pre, self.post = make_pre_post_processors(
            cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            preprocessor_overrides={"device_processor": {"device": cfg.device}},
        )
        conf = self.policy.config
        self.n_obs, self.h_eff = conf.n_obs_steps, conf.horizon - conf.n_obs_steps + 1
        ep = np.asarray(self.dataset.hf_dataset["episode_index"])
        self.ep_index = ep
        starts = np.r_[0, np.flatnonzero(np.diff(ep)) + 1]
        self.ep_start = {int(ep[s]): int(s) for s in starts}
        logging.info(
            "policy %s | scheduler=%s steps=%d | horizon=%d n_obs=%d H_eff=%d",
            cfg.policy.pretrained_path,
            conf.noise_scheduler_type,
            self.policy.diffusion.num_inference_steps,
            conf.horizon,
            self.n_obs,
            self.h_eff,
        )

    def configure(
        self,
        mode: str,
        schedule: RTCAttentionSchedule,
        jacobian: str,
        clip: str,
        r2: str = "unit",
        r2_clip: str = "relative",
    ) -> None:
        conf = self.policy.config
        conf.rtc_guidance_mode, conf.rtc_guidance_jacobian, conf.rtc_guidance_weight_clip = (
            mode,
            jacobian,
            clip,
        )
        conf.rtc_guidance_r2, conf.rtc_r2_clip = r2, r2_clip
        conf.rtc_r2_num_samples = self.cfg.r2_num_samples
        rtc = self.cfg.rtc
        conf.rtc_config = RTCConfig(
            enabled=True,
            prefix_attention_schedule=schedule,
            max_guidance_weight=rtc.max_guidance_weight,
            execution_horizon=rtc.execution_horizon,
            # Always on: per-step guidance stats are part of the output (a CLI `--rtc.*` override rebuilds
            # RTCConfig with debug=False). Latencies include the tracking overhead.
            debug=True,
            debug_maxlen=max(rtc.debug_maxlen, 100),
        )
        self.policy.init_rtc_processor()

    def obs(self, frame: int) -> dict:
        item = self.dataset[frame]
        item = {k: v for k, v in item.items() if k.startswith("observation.") or k == "task"}
        return self.pre(_batched(item))

    def chunk(self, frame: int, noise: torch.Tensor, **rtc_kwargs) -> tuple[torch.Tensor, float]:
        """Push the observation history ending at `frame` (one per control step) and predict."""
        self.policy.reset()
        ep_start = self.ep_start[int(self.ep_index[frame])]
        for j in range(max(frame - self.n_obs + 1, ep_start), frame):
            self.policy.observe(self.obs(j))
        current = self.obs(frame)
        if self.cfg.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        actions = self.policy.predict_action_chunk(current, noise=noise.clone(), **rtc_kwargs)
        if self.cfg.device.startswith("cuda"):
            torch.cuda.synchronize()
        return actions, time.perf_counter() - t0

    def run(self) -> None:
        cfg = self.cfg
        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        s, d = cfg.replan_every, cfg.inference_delay
        span = s + self.h_eff  # frames needed after k for ground truth
        ep_end = {e: int(np.flatnonzero(self.ep_index == e)[-1]) for e in self.ep_start}
        candidates = [k for k in range(len(self.ep_index)) if k + span <= ep_end[int(self.ep_index[k])]]
        rng = random.Random(cfg.seed)
        frames = sorted(rng.sample(candidates, min(cfg.num_samples, len(candidates))))

        rows, step_rows = [], []
        action_dim = self.policy.config.action_feature.shape[0]
        for si, k in enumerate(frames):
            g = torch.Generator().manual_seed(cfg.seed + si)
            noise_a = torch.randn(1, self.policy.config.horizon, action_dim, generator=g).to(cfg.device)
            noise_b = torch.randn(1, self.policy.config.horizon, action_dim, generator=g).to(cfg.device)

            self.configure("none", RTCAttentionSchedule.ZEROS, "vjp", cfg.weight_clip)
            chunk_a, _ = self.chunk(k, noise_a)
            leftover = chunk_a[:, s:]
            gt = torch.stack([self.dataset[j][ACTION] for j in range(k, k + span)])  # env units

            results = {}
            for name in cfg.variants:
                mode, schedule, jacobian, clip, r2, r2_clip = VARIANTS[name]
                self.configure(mode, schedule, jacobian, clip or cfg.weight_clip, r2, r2_clip)
                if self.policy.rtc_processor is not None:
                    self.policy.rtc_processor.reset_tracker()
                b, latency = self.chunk(k + s, noise_b, inference_delay=d, prev_chunk_left_over=leftover)
                info = self.policy.last_rtc_info
                ov, dd = info["overlap_len"], info["inference_delay"]
                results[name] = b
                prev_exec = leftover[0, dd - 1] if dd > 0 else chunk_a[0, s - 1]
                b_env = self.post(b)[0]
                fracs = [st["corr_frac"] for st in info["steps"] if "corr_frac" in st]
                rows.append(
                    {
                        "frame": k,
                        "variant": name,
                        "delay": dd,
                        "overlap_len": ov,
                        "frozen_rms_norm": float((b[0, :dd] - leftover[0, :dd]).pow(2).mean().sqrt()) if dd else float("nan"),
                        "overlap_rms_norm": float((b[0, :ov] - leftover[0, :ov]).pow(2).mean().sqrt()) if ov else float("nan"),
                        "switch_step_norm": float((b[0, dd] - prev_exec).norm()),
                        "max_d2_norm": float((b[0, 2:] - 2 * b[0, 1:-1] + b[0, :-2]).norm(dim=-1).max()),
                        "gt_rmse_env": float((b_env - gt[s : s + self.h_eff]).pow(2).mean().sqrt()),
                        # median over denoising steps of the fraction of the prefix error removed from x0_hat
                        "corr_frac_med": float(np.median(fracs)) if fracs else float("nan"),
                        "finite": bool(torch.isfinite(b).all()),
                        "latency_ms": 1e3 * latency,
                    }
                )  # fmt: skip
                for st_i, st in enumerate(info["steps"]):
                    if "grad_norm" in st:
                        step_rows.append({"frame": k, "variant": name, "step": st_i, **st})

            if si < cfg.num_plots:
                self.plot(out / f"frame_{k:06d}.png", k, chunk_a, results, gt, d)

        self.write(out, rows, step_rows)

    def plot(self, path: Path, k: int, chunk_a, results: dict, gt: torch.Tensor, d: int) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        s = self.cfg.replan_every
        a_env = self.post(chunk_a)[0].numpy()
        n_dims = a_env.shape[-1]
        fig, axes = plt.subplots(n_dims, 1, figsize=(12, 3 * n_dims), sharex=True, squeeze=False)
        for dim in range(n_dims):
            ax = axes[dim, 0]
            ax.plot(
                np.arange(len(gt)), gt[:, dim].numpy(), color="black", lw=1, ls=":", label="dataset action"
            )
            ax.plot(np.arange(self.h_eff), a_env[:, dim], color="gray", lw=2, alpha=0.6, label="chunk A")
            ax.plot(
                np.arange(s, self.h_eff), a_env[s:, dim], color="red", lw=3, alpha=0.4, label="leftover A[s:]"
            )
            for name, b in results.items():
                b_env = self.post(b)[0].numpy()
                ax.plot(
                    np.arange(s, s + self.h_eff),
                    b_env[:, dim],
                    color=COLORS.get(name),
                    lw=1.5,
                    label=f"B {name}",
                )
            ax.axvline(s, color="k", ls="--", alpha=0.4)
            ax.axvline(s + d, color="orange", ls="--", alpha=0.6)
            ax.set_ylabel(f"action[{dim}]")
            ax.grid(alpha=0.3)
        axes[0, 0].set_title(
            f"frame {k}: inference for B starts at s={s} (dashed), switch at s+d={s + d} (orange)"
        )
        axes[-1, 0].set_xlabel("control step relative to frame k")
        axes[0, 0].legend(fontsize=8, ncol=3)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        logging.info("saved %s", path)

    @staticmethod
    def write(out: Path, rows: list[dict], step_rows: list[dict]) -> None:
        with open(out / "offline_metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        if step_rows:
            keys = sorted(
                {k for r in step_rows for k in r}, key=lambda x: (x not in ("frame", "variant", "step"), x)
            )
            with open(out / "guidance_steps.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(step_rows)
        metrics = [
            "frozen_rms_norm",
            "overlap_rms_norm",
            "switch_step_norm",
            "max_d2_norm",
            "gt_rmse_env",
            "corr_frac_med",
            "latency_ms",
        ]
        lines = ["| variant | " + " | ".join(metrics) + " |", "|" + "---|" * (len(metrics) + 1)]
        for name in dict.fromkeys(r["variant"] for r in rows):
            vals = [np.nanmean([r[m] for r in rows if r["variant"] == name]) for m in metrics]
            lines.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
        table = "\n".join(lines) + "\n"
        (out / "offline_summary.md").write_text(table)
        print(table)


@parser.wrap()
def main(cfg: OfflineRTCEvalConfig) -> None:
    init_logging()
    set_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    Evaluator(cfg).run()


if __name__ == "__main__":
    main()
