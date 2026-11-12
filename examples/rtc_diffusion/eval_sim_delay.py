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

"""Deterministic simulated-latency evaluation of Real-Time Chunking for DiffusionPolicy.

No threads: inference latency is simulated as an integer number of control steps `d`.

Timeline of one asynchronous inference (methods naive / guidance_* / replacement*):
    * at step t the observation o_t is taken and inference starts (the leftover of the current chunk,
      i.e. its actions for t, t+1, ..., is passed as `prev_chunk_left_over`);
    * during steps t .. t+d-1 the robot keeps executing the previous chunk (= leftover[0:d]);
    * at step t+d the new chunk takes over at its index d (`ActionQueue.merge(..., delay=d)`);
    * the next inference starts at t + replan_every (requires replan_every >= d).
`sync`: the robot holds its last action while inference runs (d steps), then executes the new chunk from
index 0 for `replan_every` steps. With d = 0, sync == naive == the original Diffusion Policy loop.
The very first chunk of an episode is always waited for (no execution before it exists).

Every control step pushes exactly one observation into the policy (`predict_action_chunk` at inference
steps, `policy.observe` otherwise), so the n_obs_steps history is per control step.

Metrics (per episode, then aggregated per configuration):
    success, max_reward, switch discontinuity ||a_switch - a_prev|| (env units), mean step size away from
    switches, jerk (3rd finite difference), RMS mismatch between the new chunk and the leftover on the
    frozen prefix (normalized units), inference wall-clock (CUDA-synchronized) with guidance active/inactive,
    and "starved" steps (chunk exhausted -> last action held).

Example (PushT, DDPM-trained checkpoint sampled with DDIM-10):
    python examples/rtc_diffusion/eval_sim_delay.py \
        --policy.path=<migrated diffusion_pusht dir> \
        --policy.noise_scheduler_type=DDIM --policy.num_inference_steps=10 \
        --env.type=pusht --n_episodes=50 --batch_size=10 \
        --delays='[0,1,2,3,4,6]' --replan_every='[8]' --betas='[10.0]' \
        --output_dir=outputs/rtc_diffusion_sim
"""

import csv
import logging
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot import envs
from lerobot.configs import PreTrainedConfig, RTCAttentionSchedule, parser
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors, preprocess_observation
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.rtc import RTCConfig
from lerobot.utils.constants import ACTION
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

# method name -> (timing, rtc_guidance_mode, prefix schedule, guidance jacobian, weight clip or None=cfg,
#                 r2, r2 clip)
# "*_id" methods use the identity Jacobian + flow clip, i.e. exactly what LeRobot's flow RTCProcessor
# computes (pi0 / smolvla), mapped to VP variables. The others use the paper-faithful VJP.
# "*_post*" methods (experimental): VJP with r2 = per-element Gaussian posterior variance estimated from
# `r2_num_samples` unguided chunks (rtc_guidance_r2="posterior"); "_lit": beta clips 1 / r2 itself.
METHODS: dict[str, tuple[str, str, RTCAttentionSchedule, str, str | None, str, str]] = {
    "sync": ("sync", "none", RTCAttentionSchedule.ZEROS, "vjp", None, "unit", "relative"),
    "naive": ("async", "none", RTCAttentionSchedule.ZEROS, "vjp", None, "unit", "relative"),
    "guidance_zeros": ("async", "guidance", RTCAttentionSchedule.ZEROS, "vjp", None, "unit", "relative"),
    "guidance_exp": ("async", "guidance", RTCAttentionSchedule.EXP, "vjp", None, "unit", "relative"),
    "guidance_linear": ("async", "guidance", RTCAttentionSchedule.LINEAR, "vjp", None, "unit", "relative"),
    "guidance_ones": ("async", "guidance", RTCAttentionSchedule.ONES, "vjp", None, "unit", "relative"),
    "guidance_zeros_id": ("async", "guidance", RTCAttentionSchedule.ZEROS, "identity", "flow", "unit", "relative"),
    "guidance_exp_id": ("async", "guidance", RTCAttentionSchedule.EXP, "identity", "flow", "unit", "relative"),
    "guidance_exp_post": ("async", "guidance", RTCAttentionSchedule.EXP, "vjp", None, "posterior", "relative"),
    "guidance_exp_post_lit": ("async", "guidance", RTCAttentionSchedule.EXP, "vjp", None, "posterior", "literal"),
    # frozen prefix only / whole overlap
    "replacement": ("async", "replacement", RTCAttentionSchedule.ZEROS, "vjp", None, "unit", "relative"),
    "replacement_ones": ("async", "replacement", RTCAttentionSchedule.ONES, "vjp", None, "unit", "relative"),
}  # fmt: skip


@dataclass
class SimDelayEvalConfig:
    env: envs.EnvConfig
    policy: PreTrainedConfig | None = None
    output_dir: str = "outputs/rtc_diffusion_sim"
    seed: int = 1000
    n_episodes: int = 50
    batch_size: int = 10
    methods: list[str] = field(
        default_factory=lambda: [
            "sync",
            "naive",
            "guidance_zeros",
            "guidance_exp",
            "guidance_exp_id",
            "replacement",
        ]  # fmt: skip
    )
    delays: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 6])
    # Paper's execution horizon s: control steps between two inference starts.
    replan_every: list[int] = field(default_factory=lambda: [8])
    # max_guidance_weight sweep (guidance methods only).
    betas: list[float] = field(default_factory=lambda: [10.0])
    # Weight clip for VJP guidance methods ("pigdm" | "flow"); "*_id" methods always use "flow".
    weight_clip: str = "pigdm"
    # Chunks per observation for the s2 estimate of the "*_post*" methods.
    r2_num_samples: int = 16
    # LeRobot's RTCConfig.execution_horizon (overlap length). None -> H_eff (overlap = whole leftover).
    execution_horizon: int | None = None
    # Also pin raw slots [0, n_obs_steps-1) to the actually executed (normalized) actions.
    constrain_past_actions: bool = False
    max_steps: int | None = None
    rename_map: dict[str, str] = field(default_factory=dict)
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        policy_path = parser.get_path_arg("policy")
        if not policy_path:
            raise ValueError("--policy.path is required")
        cli_overrides = parser.get_cli_overrides("policy") or []
        self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
        self.policy.pretrained_path = Path(policy_path)
        unknown = [m for m in self.methods if m not in METHODS]
        if unknown:
            raise ValueError(f"Unknown methods {unknown}; choose from {list(METHODS)}")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


# ------------------------------------------------------------------------------------------------
# One batched rollout
# ------------------------------------------------------------------------------------------------


def _successes(info: dict, num_envs: int) -> list[bool]:
    if "final_info" in info:
        return info["final_info"]["is_success"].tolist()
    if "is_success" in info:
        s = info["is_success"]
        return s.tolist() if hasattr(s, "tolist") else [bool(s)] * num_envs
    return [False] * num_envs


def _task(env, num_envs: int) -> list[str]:
    for attr in ("task_description", "task"):
        try:
            return list(env.call(attr))
        except (AttributeError, NotImplementedError):
            continue
    return [""] * num_envs


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_batch(
    env,
    policy,
    procs: tuple,
    *,
    timing: str,
    delay: int,
    replan_every: int,
    seeds: list[int],
    max_steps: int,
    execution_horizon: int,
    constrain_past_actions: bool = False,
) -> tuple[list[dict[str, Any]], list[tuple[float, bool]]]:
    """Roll out one batch of envs under a fixed (method, d, replan_every) schedule."""
    preprocessor, postprocessor, env_preprocessor, env_postprocessor = procs
    device = next(policy.parameters()).device
    num_envs = env.num_envs
    h_eff = policy.config.horizon - policy.config.n_obs_steps + 1

    def process(raw_obs) -> dict:
        obs = preprocess_observation(raw_obs)
        obs["task"] = tasks
        return preprocessor(env_preprocessor(obs))

    def to_env(action_env_units: torch.Tensor) -> np.ndarray:
        return env_postprocessor({ACTION: action_env_units})[ACTION].to("cpu").numpy()

    timings: list[tuple[float, bool]] = []

    def infer(obs: dict, leftover: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        _sync(device)
        t0 = time.perf_counter()
        n_past = policy.config.n_obs_steps - 1
        past = (
            torch.stack(executed_norm[-n_past:], dim=1)
            if constrain_past_actions and n_past > 0 and executed_norm
            else None
        )
        chunk = policy.predict_action_chunk(
            obs,
            inference_delay=delay,
            prev_chunk_left_over=leftover,
            execution_horizon=execution_horizon,
            prev_executed_actions=past,
        )
        _sync(device)
        timings.append((time.perf_counter() - t0, bool(policy.last_rtc_info["active"])))
        return chunk, postprocessor(chunk)

    policy.reset()
    raw_obs, _ = env.reset(seed=seeds)
    tasks = _task(env, num_envs)

    executed, rewards, successes, dones = [], [], [], []
    switch_steps: list[int] = []
    frozen_err: list[float] = []  # RMS(new[:d] - leftover[:d]) in normalized units, per switch
    starved = np.zeros(num_envs, dtype=int)
    done = np.zeros(num_envs, dtype=bool)

    cur_norm = cur_env = None  # current chunk (B, H_eff, A) normalized / env units
    cur_t0 = 0  # time of the current chunk's executable index 0
    pending = None  # (norm, env, t_ready, t0)
    next_infer = 0
    hold_until = -1  # sync: hold last action until this step (exclusive)
    last_action: np.ndarray | None = None
    last_action_norm: torch.Tensor | None = None
    executed_norm: list[torch.Tensor] = []  # normalized executed actions (B, A), for past-slot pinning
    t = 0
    while t < max_steps and not done.all():
        obs = process(raw_obs)
        # A chunk whose inference started at t - d takes over now. This must happen *before* a new
        # inference starting at t computes its leftover (case d == replan_every).
        if pending is not None and t == pending[2]:
            cur_norm, cur_env, _, cur_t0 = pending
            pending = None
            switch_steps.append(t)
        pushed = False
        if t == next_infer:
            if cur_norm is None:  # first chunk: blocking, no delay
                cur_norm, cur_env = infer(obs, None)
                cur_t0 = t
                next_infer = t + replan_every
            elif timing == "sync":
                cur_norm, cur_env = infer(obs, None)
                cur_t0 = t + delay  # executes from index 0 after holding for `delay` steps
                hold_until = t + delay
                if delay == 0:
                    switch_steps.append(t)
                next_infer = t + delay + replan_every
            else:
                leftover = cur_norm[:, t - cur_t0 :]
                new_norm, new_env = infer(obs, leftover if leftover.shape[1] > 0 else None)
                if delay > 0 and leftover.shape[1] > 0:
                    n = min(delay, leftover.shape[1])
                    frozen_err.append(float((new_norm[:, :n] - leftover[:, :n]).pow(2).mean().sqrt()))
                pending = (new_norm, new_env, t + delay, t)
                next_infer = t + replan_every
            pushed = True
        if not pushed:
            policy.observe(obs)

        if timing == "sync" and t == hold_until and delay > 0:
            switch_steps.append(t)
        if pending is not None and t == pending[2]:  # d == 0: the new chunk is used immediately
            cur_norm, cur_env, _, cur_t0 = pending
            pending = None
            switch_steps.append(t)

        if timing == "sync" and t < hold_until:
            action, action_norm = last_action, last_action_norm  # robot waits during inference
        else:
            idx = t - cur_t0
            if idx < h_eff:
                action, action_norm = to_env(cur_env[:, idx]), cur_norm[:, idx]
            else:  # chunk exhausted before the next one is ready
                action, action_norm = last_action, last_action_norm
                starved += ~done
        raw_obs, reward, terminated, truncated, info = env.step(action)
        last_action, last_action_norm = action, action_norm
        executed_norm.append(action_norm)

        executed.append(action.copy())
        rewards.append(reward)
        successes.append(_successes(info, num_envs))
        done = terminated | truncated | done
        if t + 1 == max_steps:
            done[:] = True
        dones.append(done.copy())
        t += 1

    actions = np.stack(executed, axis=1)  # (B, T, A)
    rewards_arr, succ_arr, done_arr = np.stack(rewards, 1), np.array(successes).T, np.stack(dones, 1)
    results = []
    for b in range(num_envs):
        t_end = int(np.argmax(done_arr[b])) + 1 if done_arr[b].any() else actions.shape[1]
        a = actions[b, :t_end]
        sw = [s for s in switch_steps if 1 <= s < t_end]
        step = np.linalg.norm(np.diff(a, axis=0), axis=-1)  # step[k] = ||a[k+1] - a[k]||
        disc = [step[s - 1] for s in sw]
        non_switch = np.delete(step, [s - 1 for s in sw]) if len(step) else step
        jerk = (
            np.linalg.norm(a[3:] - 3 * a[2:-1] + 3 * a[1:-2] - a[:-3], axis=-1) if t_end > 3 else np.zeros(0)
        )
        jerk_sw = [jerk[max(s - 3, 0) : s].max() for s in sw if s >= 3 and len(jerk[max(s - 3, 0) : s])]
        results.append(
            {
                "seed": seeds[b],
                "success": bool(succ_arr[b, :t_end].any()),
                "max_reward": float(rewards_arr[b, :t_end].max()),
                "sum_reward": float(rewards_arr[b, :t_end].sum()),
                "steps": t_end,
                "n_switches": len(sw),
                "switch_disc": float(np.mean(disc)) if disc else float("nan"),
                "nonswitch_step": float(np.mean(non_switch)) if len(non_switch) else float("nan"),
                "jerk_mean": float(jerk.mean()) if len(jerk) else float("nan"),
                "jerk_switch": float(np.mean(jerk_sw)) if jerk_sw else float("nan"),
                "frozen_rms_norm": float(np.mean(frozen_err)) if frozen_err else float("nan"),
                "starved_steps": int(starved[b]),
            }
        )
    return results, timings


# ------------------------------------------------------------------------------------------------
# Sweep
# ------------------------------------------------------------------------------------------------


def _nanmean(xs: list[float]) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    return float(np.mean(xs)) if xs else float("nan")


def summarize(rows: list[dict], timings: list[tuple[float, bool]]) -> dict[str, Any]:
    guided = [t for t, active in timings if active]
    plain = [t for t, active in timings if not active]
    return {
        "n_episodes": len(rows),
        "success_rate": _nanmean([float(r["success"]) for r in rows]),
        "max_reward": _nanmean([r["max_reward"] for r in rows]),
        "switch_disc": _nanmean([r["switch_disc"] for r in rows]),
        "nonswitch_step": _nanmean([r["nonswitch_step"] for r in rows]),
        "jerk_mean": _nanmean([r["jerk_mean"] for r in rows]),
        "jerk_switch": _nanmean([r["jerk_switch"] for r in rows]),
        "frozen_rms_norm": _nanmean([r["frozen_rms_norm"] for r in rows]),
        "starved_steps": int(sum(r["starved_steps"] for r in rows)),
        "latency_ms_guided": 1e3 * statistics.median(guided) if guided else float("nan"),
        "latency_ms_unguided": 1e3 * statistics.median(plain) if plain else float("nan"),
        "n_guided_calls": len(guided),
    }


def write_markdown(rows: list[dict], path: Path, keys: list[str]) -> str:
    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(fmt(r[k]) for k in keys) + " |" for r in rows]
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


@parser.wrap()
def main(cfg: SimDelayEvalConfig) -> None:
    init_logging()
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(cfg.seed)

    env_dict = make_env(cfg.env, n_envs=cfg.batch_size, trust_remote_code=cfg.trust_remote_code)
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)
    procs = (preprocessor, postprocessor, env_preprocessor, env_postprocessor)

    h_eff = policy.config.horizon - policy.config.n_obs_steps + 1
    execution_horizon = h_eff if cfg.execution_horizon is None else cfg.execution_horizon
    logging.info(
        "policy: %s, scheduler=%s, steps=%s, horizon=%d, n_obs_steps=%d, H_eff=%d",
        cfg.policy.pretrained_path,
        policy.config.noise_scheduler_type,
        policy.diffusion.num_inference_steps,
        policy.config.horizon,
        policy.config.n_obs_steps,
        h_eff,
    )

    configs = []
    for method in cfg.methods:
        mode = METHODS[method][1]
        for s in cfg.replan_every:
            for d in cfg.delays:
                if d > s:
                    logging.warning("skip %s d=%d > replan_every=%d (overlapping inferences)", method, d, s)
                    continue
                if s + d > h_eff:
                    logging.warning(
                        "%s d=%d s=%d: s + d > H_eff=%d, chunks will run out", method, d, s, h_eff
                    )
                for beta in cfg.betas if mode == "guidance" else [float("nan")]:
                    configs.append((method, d, s, beta))

    episode_rows, summary_rows = [], []
    episode_csv = out / "episodes.csv"
    for ci, (method, d, s, beta) in enumerate(configs):
        timing, mode, schedule, jacobian, clip, r2, r2_clip = METHODS[method]
        clip = clip or cfg.weight_clip
        policy.config.rtc_guidance_mode = mode
        policy.config.rtc_guidance_weight_clip = clip
        policy.config.rtc_guidance_jacobian = jacobian
        policy.config.rtc_guidance_r2, policy.config.rtc_r2_clip = r2, r2_clip
        policy.config.rtc_r2_num_samples = cfg.r2_num_samples
        policy.config.rtc_constrain_past_actions = cfg.constrain_past_actions
        policy.config.rtc_config = RTCConfig(
            enabled=True,
            prefix_attention_schedule=schedule,
            max_guidance_weight=beta if not math.isnan(beta) else 10.0,
            execution_horizon=execution_horizon,
        )
        policy.init_rtc_processor()
        cfg_cols = {
            "method": method,
            "delay": d,
            "replan_every": s,
            "beta": beta,
            "weight_clip": clip if mode == "guidance" else "",
            "jacobian": jacobian if mode == "guidance" else "",
            "r2": r2 if mode == "guidance" else "",
            "r2_clip": r2_clip if mode == "guidance" and r2 != "unit" else "",
        }

        rows, timings = [], []
        for suite, tasks in env_dict.items():
            for task_id, env in tasks.items():
                max_steps = cfg.max_steps or env.call("_max_episode_steps")[0]
                n_batches = math.ceil(cfg.n_episodes / env.num_envs)
                for bi in range(n_batches):
                    set_seed(cfg.seed + bi)  # same policy RNG stream for every configuration
                    seeds = list(range(cfg.seed + bi * env.num_envs, cfg.seed + (bi + 1) * env.num_envs))
                    with torch.no_grad():
                        res, tim = run_batch(
                            env,
                            policy,
                            procs,
                            timing=timing,
                            delay=d,
                            replan_every=s,
                            seeds=seeds,
                            max_steps=max_steps,
                            execution_horizon=execution_horizon,
                            constrain_past_actions=cfg.constrain_past_actions,
                        )
                    for r in res[: cfg.n_episodes - len(rows)]:
                        rows.append({**cfg_cols, "suite": suite, "task_id": task_id, **r})
                    timings += tim
        summary = {**cfg_cols, **summarize(rows, timings)}
        summary_rows.append(summary)
        episode_rows += rows
        logging.info(
            "[%d/%d] %s d=%d s=%d beta=%s: success=%.3f max_reward=%.3f switch_disc=%.2f jerk=%.2f "
            "frozen_rms=%.3f latency(ms) guided=%.1f unguided=%.1f",
            ci + 1,
            len(configs),
            method,
            d,
            s,
            beta,
            summary["success_rate"],
            summary["max_reward"],
            summary["switch_disc"],
            summary["jerk_mean"],
            summary["frozen_rms_norm"],
            summary["latency_ms_guided"],
            summary["latency_ms_unguided"],
        )
        # Write incrementally so partial sweeps are not lost.
        with open(episode_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(episode_rows[0]))
            writer.writeheader()
            writer.writerows(episode_rows)
        with open(out / "summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)

    keys = [
        "method", "delay", "replan_every", "beta", "n_episodes", "success_rate", "max_reward", "switch_disc",
        "nonswitch_step", "jerk_mean", "jerk_switch", "frozen_rms_norm", "starved_steps",
        "latency_ms_guided", "latency_ms_unguided",
    ]  # fmt: skip
    print(write_markdown(summary_rows, out / "summary.md", keys))
    with open(out / "run_config.txt", "w") as f:
        f.write(repr(asdict(cfg)))
    close_envs(env_dict)


if __name__ == "__main__":
    main()
