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

"""Real-Time Chunking (RTC) for DiffusionPolicy (DDPM / DDIM, epsilon or sample prediction).

RTC (Black et al., 2025, arXiv:2506.07339) is an inference-time method for asynchronous action
chunking: while a new chunk is generated, the part of the previous chunk that will be executed
anyway (the "prefix") is enforced with PiGDM-style inpainting guidance. Training is unchanged.

The LeRobot RTC infrastructure (`lerobot.policies.rtc`) targets flow-matching policies. This module
implements the diffusion-native (VP / DDPM parameterization) counterpart and plugs it into
`DiffusionModel.conditional_sample` through the `rtc_guide` hook.

Coordinates (the most error-prone part)
---------------------------------------
* The diffusion model predicts `horizon` actions. Raw index `j` is the action at time
  `t - (n_obs_steps - 1) + j` where `t` is the time of the latest observation in the snapshot.
* Executable coordinate `i` (what `predict_action_chunk` returns in RTC mode) maps to raw index
  `n_obs_steps - 1 + i` (see `exec_to_raw_index`); there are `H_eff = horizon - n_obs_steps + 1`
  executable steps.
* `prev_chunk_left_over[i]` is the previous chunk's action for time `t + i`, i.e. it is aligned
  with executable coordinate `i` of the new chunk (same convention as `ActionQueue.get_left_over`).
* Both the leftover and the guidance target live in the policy's *normalized* action space
  (model output before the postprocessor). `ActionQueue.merge(original, processed, delay)` must
  receive the normalized chunk as `original`.

Naming: LeRobot's `RTCConfig.execution_horizon` is the length of the region kept consistent with
the previous chunk (the paper's `H - s`). It is called `overlap_len` here. The paper's execution
horizon `s` (steps executed between inference starts) is called `replan_every` in the eval scripts.

Guided sampling (mode "guidance"), per denoising step t, with a_t = alphas_cumprod[t]:
    x0_hat = (x_t - sqrt(1 - a_t) * eps) / sqrt(a_t)        (epsilon)   |   x0_hat = out  (sample)
    e      = (Y - x0_hat) * W                                (W: raw-horizon-aligned prefix weights)
    g      = VJP(x0_hat wrt x_t, e)
    w      = min(beta, 1 / r2),  r2 = 1 - a_t                (r2 = Var[x0 | x_t] for a N(0, I) prior)
    eps'   = eps - sqrt(1 - a_t) * w * g                     (epsilon)
    x0'    = out + (1 - a_t) / sqrt(a_t) * w * g             (sample; identical update)
    x_{t-1} = scheduler.step(out', t, x_t)

Relation to the flow-matching RTCProcessor: with the VP <-> rectified-flow change of variables
(`sqrt(a) = tau / s`, `sqrt(1 - a) = (1 - tau) / s`, `s = sqrt(tau^2 + (1 - tau)^2)`), the implied
update of x0_hat is identical (`g / sqrt(a_t)`) whenever beta does not clip. The flow weight
`min(beta, (1 - tau) / (tau r_tau^2))` equals `min(beta, 1 / sqrt(a (1 - a)))`, which clips at *both*
ends, while `min(beta, 1 / r2)` only clips the low-noise end. `rtc_guidance_weight_clip="flow"`
reproduces the flow clipping exactly (see `guidance_weight`); "pigdm" (default) is the formula above.

NOTE on LeRobot's flow `RTCProcessor.denoise_step`: it evaluates the velocity *before*
`x_t.requires_grad_(True)`, so its "VJP" is taken with an identity Jacobian (correction == err). The
paper-faithful VJP is the default here (`rtc_guidance_jacobian="vjp"`). `"identity"` reproduces the
LeRobot behaviour in VP variables (`g = s * e`, `s = 1 / (sqrt(a) + sqrt(1 - a))`) without a backward
pass. For policies with a narrow conditional action distribution (e.g. PushT DP) the true Jacobian of
x0_hat is ~0 and VJP guidance is weak unless beta is very large.

Assumptions that were NOT verified empirically and are documented as such:
* (A1) r2 = 1 - a_t assumes a unit-variance data prior (the PiGDM / RTC choice). With MIN_MAX
  normalized actions the true variance is < 1, so guidance is somewhat weaker than "exact".
  `rtc_guidance_r2="posterior"` (experimental) replaces it by the per-element Gaussian posterior variance
  `s2 (1 - a) / (a s2 + 1 - a)`, with `s2` estimated from K unguided chunks for the same observation, and
  applies `1 / r2_d` to the cotangent before the VJP. For an independent Gaussian prior with variance s2 this
  gives `J / r2 = sqrt(a) / (1 - a)` and, unclipped, moves x0_hat exactly onto the target. Correlations
  (across time / action dims) and multimodality of the true action distribution are ignored.
  Measured on PushT DP (examples/rtc_diffusion/diag_jacobian_r2.py): this r2 matches the model's own posterior
  variance, but the learned Jacobian does not satisfy J / r2 = sqrt(a) / (1 - a) (random sign, 2-100x off,
  strongly asymmetric). The variant therefore diverges unless beta clips 1 / r2 ("literal").
* (A2) The default `max_guidance_weight` (beta) comes from the flow-matching parameterization. With
  the "pigdm" clip it only clips `1 / (1 - a_t)`, i.e. it matters for the last (low-noise) steps.
  It must be swept.
* (A3) With DDPM and many steps, the first steps have a_t ~ 1e-7, which amplifies the Jacobian of
  x0_hat by 1 / sqrt(a_t). Use DDIM with few steps (e.g. 10) for guidance.
* (A4) `inference_delay=None` is treated as 0 (pi0's processor would fail on None).
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from ..rtc.modeling_rtc import RTCProcessor
from ..utils import populate_queues

if TYPE_CHECKING:
    from .modeling_diffusion import DiffusionModel, DiffusionPolicy

logger = logging.getLogger(__name__)

# Guidance adds one VJP per denoising step; warn above this many steps.
MANY_GUIDED_STEPS_WARNING_THRESHOLD = 20

SUPPORTED_PREDICTION_TYPES = ("epsilon", "sample")

# One process-wide lock guarding the observation queues of diffusion policies. A lock stored on the
# policy would break `copy.deepcopy(policy)`; contention is negligible (pushes are O(1)).
_OBS_QUEUE_LOCK = threading.RLock()


# ---------------------------------------------------------------------------------------------
# Index mapping (single source of truth)
# ---------------------------------------------------------------------------------------------


def executable_horizon(horizon: int, n_obs_steps: int) -> int:
    """Number of executable actions per chunk, `H_eff = horizon - n_obs_steps + 1`."""
    h_eff = horizon - n_obs_steps + 1
    if h_eff <= 0:
        raise ValueError(f"horizon ({horizon}) must be >= n_obs_steps ({n_obs_steps}).")
    return h_eff


def exec_to_raw_index(i: int, n_obs_steps: int) -> int:
    """Executable coordinate `i` (== `prev_chunk_left_over[i]`, time `t + i`) -> raw horizon index."""
    return n_obs_steps - 1 + i


def resolve_overlap(
    execution_horizon: int, leftover_len: int, h_eff: int, inference_delay: int | None
) -> tuple[int, int]:
    """Clamp the overlap length and the inference delay.

    Returns:
        overlap_len: `min(execution_horizon, leftover_len, h_eff)` (LeRobot's `execution_horizon`).
        inference_delay: clamped to `[0, overlap_len]`.
    """
    overlap_len = max(0, min(int(execution_horizon), int(leftover_len), int(h_eff)))
    delay = 0 if inference_delay is None else max(0, int(inference_delay))
    return overlap_len, min(delay, overlap_len)


def build_raw_prefix(
    exec_weights: Tensor,
    leftover: Tensor | None,
    *,
    horizon: int,
    n_obs_steps: int,
    batch_size: int,
    action_dim: int,
    past_actions: Tensor | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor]:
    """Place executable-coordinate targets/weights onto the raw horizon.

    Args:
        exec_weights: (H_eff,) prefix weights in executable coordinates.
        leftover: (B, L, A) previous-chunk actions in executable coordinates (normalized), or None.
        past_actions: optional (B, K, A) actually executed actions ending at time `t - 1`
            (the last row is time `t - 1`). Rows are placed on raw indices `[0, n_obs_steps - 1)`,
            right-aligned, with weight 1.

    Returns:
        target: (B, horizon, A) guidance target `Y` (zeros where the weight is 0).
        weights: (horizon,) weights `W`.
    """
    h_eff = executable_horizon(horizon, n_obs_steps)
    if exec_weights.shape != (h_eff,):
        raise ValueError(f"exec_weights must have shape ({h_eff},), got {tuple(exec_weights.shape)}")

    start = exec_to_raw_index(0, n_obs_steps)
    target = torch.zeros(batch_size, horizon, action_dim, dtype=dtype, device=device)
    weights = torch.zeros(horizon, dtype=dtype, device=device)

    if leftover is not None:
        n = min(leftover.shape[1], h_eff)
        target[:, start : start + n] = leftover[:, :n].to(dtype=dtype, device=device)
        weights[start : start + n] = exec_weights[:n].to(dtype=dtype, device=device)

    if past_actions is not None and start > 0:
        k = min(past_actions.shape[1], start)
        if k > 0:
            target[:, start - k : start] = past_actions[:, -k:].to(dtype=dtype, device=device)
            weights[start - k : start] = 1.0

    return target, weights


# ---------------------------------------------------------------------------------------------
# Guidance math (pure functions, tested independently)
# ---------------------------------------------------------------------------------------------


def _check_prediction_type(prediction_type: str) -> None:
    if prediction_type not in SUPPORTED_PREDICTION_TYPES:
        raise ValueError(
            f"RTC for DiffusionPolicy supports prediction_type in {SUPPORTED_PREDICTION_TYPES}, "
            f"got {prediction_type!r}."
        )


def tweedie_x0(model_output: Tensor, x_t: Tensor, alpha_bar: float, prediction_type: str) -> Tensor:
    """Posterior-mean estimate of x0 from the model output (no clipping, so gradients survive)."""
    _check_prediction_type(prediction_type)
    if prediction_type == "epsilon":
        return (x_t - (1.0 - alpha_bar) ** 0.5 * model_output) / alpha_bar**0.5
    return model_output


def guidance_weight(alpha_bar: float, max_guidance_weight: float, clip: str = "pigdm") -> float:
    """Guidance weight `w` multiplying the VJP `g`.

    "pigdm": `w = min(beta, 1 / r2)`, `r2 = 1 - alpha_bar`.
    "flow":  `w = sqrt(a / (1 - a)) * min(beta, 1 / sqrt(a (1 - a)))`, i.e. the LeRobot flow-matching
             weight `min(beta, (1 - tau) / (tau r_tau^2))` expressed in VP variables. Unclipped, both
             reduce to `1 / (1 - a)`.
    """
    beta = float(max_guidance_weight)
    r2 = 1.0 - alpha_bar
    if clip == "pigdm":
        return beta if r2 <= 0.0 else float(min(beta, 1.0 / r2))
    if clip == "flow":
        if r2 <= 0.0 or alpha_bar <= 0.0:
            raise ValueError(f"alpha_bar must be in (0, 1) for the flow clip, got {alpha_bar}")
        return float((alpha_bar / r2) ** 0.5 * min(beta, 1.0 / (alpha_bar * r2) ** 0.5))
    raise ValueError(f"Unknown guidance weight clip {clip!r}")


def posterior_r2(alpha_bar: float, prior_var: Tensor) -> Tensor:
    """`Var[x0_d | x_t]` for independent Gaussian dims `x0_d ~ N(mu_d, s2_d)`: `s2 (1 - a) / (a s2 + 1 - a)`.

    For this prior the Tweedie Jacobian is `d x0_hat_d / d x_t,d = sqrt(a) s2 / (a s2 + 1 - a)`, so
    `J / r2 = sqrt(a) / (1 - a)` exactly, independent of `s2`. With `s2 = 1` this is the unit prior.
    """
    return prior_var * (1.0 - alpha_bar) / (alpha_bar * prior_var + (1.0 - alpha_bar))


def posterior_guidance_weights(
    alpha_bar: float,
    prior_var: Tensor,
    max_guidance_weight: float,
    clip: str = "pigdm",
    r2_clip: str = "relative",
) -> Tensor:
    """Per-dimension guidance weights `w_d` (same shape as `prior_var`) for `r2_d = posterior_r2(...)`.

    "relative": `w_d = guidance_weight(a, beta, clip) * (1 - a) / r2_d` (beta clips the unit-prior weight,
                the posterior factor is applied on top; beta keeps its meaning from the unit-prior case).
    "literal":  `clip` applied to `1 / r2_d` directly ("pigdm": `min(beta, 1 / r2_d)`; "flow":
                `sqrt(a / (1 - a)) * min(beta, sqrt((1 - a) / a) / r2_d)`).
    Unclipped, both are `1 / r2_d`.
    """
    beta = float(max_guidance_weight)
    r2 = posterior_r2(alpha_bar, prior_var)
    if r2_clip == "relative":
        return guidance_weight(alpha_bar, beta, clip) * (1.0 - alpha_bar) / r2
    if r2_clip == "literal":
        if clip == "pigdm":
            return torch.clamp(1.0 / r2, max=beta)
        if clip == "flow":
            c = ((1.0 - alpha_bar) / alpha_bar) ** 0.5
            return torch.clamp(c / r2, max=beta) / c
        raise ValueError(f"Unknown guidance weight clip {clip!r}")
    raise ValueError(f"Unknown r2 clip {r2_clip!r}")


def apply_guidance_correction(
    model_output: Tensor, grad: Tensor, alpha_bar: float, weight: float, prediction_type: str
) -> Tensor:
    """Correct the model output with the guidance gradient `g` (epsilon and sample are equivalent)."""
    _check_prediction_type(prediction_type)
    if prediction_type == "epsilon":
        return model_output - (1.0 - alpha_bar) ** 0.5 * weight * grad
    return model_output + (1.0 - alpha_bar) / alpha_bar**0.5 * weight * grad


def flow_scale(alpha_bar: float) -> float:
    """`s` with `x_flow = s * x_vp` for the VP <-> rectified-flow change of variables."""
    return 1.0 / (alpha_bar**0.5 + (1.0 - alpha_bar) ** 0.5)


def previous_timestep(scheduler: Any, timestep: int | Tensor) -> int:
    """Previous timestep exactly as the scheduler's `step` computes it (-1 means "final")."""
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    t = int(timestep)
    if isinstance(scheduler, DDIMScheduler):
        return t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    if isinstance(scheduler, DDPMScheduler):
        return int(scheduler.previous_timestep(timestep))
    raise TypeError(f"Unsupported noise scheduler for RTC: {type(scheduler).__name__}")


# ---------------------------------------------------------------------------------------------
# Guided denoising loop
# ---------------------------------------------------------------------------------------------


class DiffusionRTCGuide:
    """Per-call RTC state passed to `DiffusionModel.conditional_sample(..., rtc_guide=...)`.

    If the guide is inactive (mode "none", or all weights zero) the loop performs exactly the same
    operations as the original sampler (one U-Net forward per step), so outputs are bit-identical.
    """

    def __init__(
        self,
        *,
        mode: str,
        target: Tensor,
        weights: Tensor,
        max_guidance_weight: float,
        prediction_type: str,
        inference_delay: int = 0,
        overlap_len: int = 0,
        processor: RTCProcessor | None = None,
        weight_clip: str = "pigdm",
        jacobian: str = "vjp",
        prior_var: Tensor | None = None,
        r2_clip: str = "relative",
    ):
        if mode not in ("none", "guidance", "replacement"):
            raise ValueError(f"Unknown RTC mode {mode!r}")
        if target.ndim != 3 or weights.ndim != 1 or target.shape[1] != weights.shape[0]:
            raise ValueError(
                f"Expected target (B, horizon, A) and weights (horizon,), got {tuple(target.shape)} "
                f"and {tuple(weights.shape)}"
            )
        _check_prediction_type(prediction_type)
        self.mode = mode
        self.target = target
        self.weights = weights
        self.max_guidance_weight = float(max_guidance_weight)
        self.prediction_type = prediction_type
        self.inference_delay = int(inference_delay)
        self.overlap_len = int(overlap_len)
        self.processor = processor
        self.weight_clip = weight_clip
        guidance_weight(0.5, self.max_guidance_weight, weight_clip)  # validate `weight_clip` early
        if jacobian not in ("vjp", "identity"):
            raise ValueError(f"Unknown RTC guidance jacobian {jacobian!r}")
        self.jacobian = jacobian
        # Per-element prior variance s2 (B, horizon, A) -> r2 = posterior_r2(a, s2) instead of 1 - a.
        if prior_var is not None and prior_var.shape != target.shape:
            raise ValueError(f"prior_var shape {tuple(prior_var.shape)} != target {tuple(target.shape)}")
        self.prior_var = prior_var
        if r2_clip not in ("relative", "literal"):
            raise ValueError(f"Unknown r2 clip {r2_clip!r}")
        self.r2_clip = r2_clip
        self.step_stats: list[dict[str, float]] = []

    @property
    def active(self) -> bool:
        return self.mode != "none" and bool((self.weights > 0).any())

    @property
    def debug(self) -> bool:
        return self.processor is not None and self.processor.is_debug_enabled()

    # -- main entry point ------------------------------------------------------------------------

    def denoise(
        self,
        model: DiffusionModel,
        sample: Tensor,
        global_cond: Tensor | None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Run the reverse process. `scheduler.set_timesteps` must already have been called."""
        if self.target.shape[0] != sample.shape[0] or self.target.shape[1:] != sample.shape[1:]:
            raise ValueError(
                f"RTC target shape {tuple(self.target.shape)} does not match sample {tuple(sample.shape)}"
            )
        scheduler = model.noise_scheduler
        if self.active and self.mode == "guidance" and self.jacobian == "identity":
            for t in scheduler.timesteps:
                sample = self._guided_step(model, sample, global_cond, t, generator)
            return sample
        if self.active and self.mode == "guidance":
            # inference_mode(False): the caller may run under torch.inference_mode (e.g. lerobot-eval).
            # Inputs are cloned so that no inference tensor is saved for backward.
            with torch.inference_mode(False):
                sample = sample.clone()
                global_cond = None if global_cond is None else global_cond.detach().clone()
                for t in scheduler.timesteps:
                    sample = self._guided_step(model, sample, global_cond, t, generator)
            return sample

        replace = self.active and self.mode == "replacement"
        num_steps = len(scheduler.timesteps)
        for i, t in enumerate(scheduler.timesteps):
            # Same operations as the original `conditional_sample` loop.
            model_output = model.unet(
                sample,
                torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                global_cond=global_cond,
            )
            if self.debug:
                self._track(scheduler, t, x_t=sample, v_t=model_output)
            sample = scheduler.step(model_output, t, sample, generator=generator).prev_sample
            if replace:
                sample = self._replace_known(
                    scheduler, sample, t, is_last=i == num_steps - 1, generator=generator
                )
        return sample

    # -- guidance ------------------------------------------------------------------------------

    def _guided_step(
        self,
        model: DiffusionModel,
        sample: Tensor,
        global_cond: Tensor | None,
        t: Tensor,
        generator: torch.Generator | None,
    ) -> Tensor:
        scheduler = model.noise_scheduler
        alpha_bar = float(scheduler.alphas_cumprod[int(t)])
        timestep = torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device)
        weights = self.weights.to(dtype=sample.dtype)[None, :, None]
        target = self.target.to(dtype=sample.dtype)
        # Scalar weight (r2 = 1 - a) applied after the VJP, or per-element weights 1 / r2_d applied to the
        # cotangent *before* the VJP: grad log N(Y; x0_hat, diag(r2)) = J^T (W e / r2).
        lam = None
        if self.prior_var is not None:
            lam = posterior_guidance_weights(
                alpha_bar,
                self.prior_var.to(dtype=sample.dtype, device=sample.device),
                self.max_guidance_weight,
                self.weight_clip,
                self.r2_clip,
            )
        w = guidance_weight(alpha_bar, self.max_guidance_weight, self.weight_clip) if lam is None else 1.0

        if self.jacobian == "vjp":
            with torch.enable_grad():
                # Only `x_t` is an autograd input: model parameters never receive `.grad`.
                x_t = sample.detach().requires_grad_(True)
                out = model.unet(x_t, timestep, global_cond=global_cond)
                x0_hat = tweedie_x0(out, x_t, alpha_bar, self.prediction_type)
                err = (target - x0_hat.detach()) * weights
                cot = err if lam is None else err * lam
                (grad,) = torch.autograd.grad(x0_hat, x_t, grad_outputs=cot, retain_graph=False)
        else:
            # "identity": d x1_hat / d x_flow := I (LeRobot RTCProcessor behaviour), i.e. g = s * e in VP
            # variables. No backward pass.
            out = model.unet(sample, timestep, global_cond=global_cond)
            x0_hat = tweedie_x0(out, sample, alpha_bar, self.prediction_type)
            err = (target - x0_hat) * weights
            grad = flow_scale(alpha_bar) * (err if lam is None else err * lam)

        out = out.detach()
        out_guided = apply_guidance_correction(out, grad, alpha_bar, w, self.prediction_type)

        if self.debug:
            mask = (self.weights > 0)[None, :, None].expand_as(err)
            self._track(
                scheduler,
                t,
                x_t=sample,
                v_t=out,
                x1_t=x0_hat.detach(),
                correction=grad,
                err=err,
                weights=self.weights,
                guidance_weight=w if lam is None else float(lam[mask].mean()) if mask.any() else 0.0,
                alpha_bar=alpha_bar,
                delta_out=out_guided - out,
                # Implied change of x0_hat (identical for epsilon / sample parameterizations).
                delta_x0=(1.0 - alpha_bar) / alpha_bar**0.5 * w * grad,
            )

        return scheduler.step(out_guided, t, sample.detach(), generator=generator).prev_sample

    # -- replacement (RePaint / Diffuser-style hard inpainting) ---------------------------------

    def _replace_known(
        self,
        scheduler: Any,
        sample: Tensor,
        t: Tensor,
        *,
        is_last: bool,
        generator: torch.Generator | None,
    ) -> Tensor:
        """Overwrite the W==1 slots of `x_{t-1}` with a sample of `q(x_{t-1} | Y)` (Y at the end)."""
        from diffusers.utils.torch_utils import randn_tensor

        mask = (self.weights >= 1.0)[None, :, None].to(sample.device)
        target = self.target.to(dtype=sample.dtype, device=sample.device)
        prev_t = -1 if is_last else previous_timestep(scheduler, t)
        if prev_t < 0:
            known = target
        else:
            alpha_bar_prev = float(scheduler.alphas_cumprod[prev_t])
            noise = randn_tensor(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
            known = alpha_bar_prev**0.5 * target + (1.0 - alpha_bar_prev) ** 0.5 * noise
        return torch.where(mask, known, sample)

    # -- debug ----------------------------------------------------------------------------------

    def _track(
        self,
        scheduler: Any,
        t: Tensor,
        *,
        x_t: Tensor,
        v_t: Tensor,
        x1_t: Tensor | None = None,
        correction: Tensor | None = None,
        err: Tensor | None = None,
        weights: Tensor | None = None,
        guidance_weight: float | None = None,
        alpha_bar: float | None = None,
        delta_out: Tensor | None = None,
        delta_x0: Tensor | None = None,
    ) -> None:
        """Record per-step info in the shared RTC debug tracker format.

        The tracker key `time` is `t / num_train_timesteps` (1 = noise, 0 = data), matching the
        LeRobot flow convention. Diffusion-specific scalars go in `metadata`.
        """
        timestep = int(t)
        num_train = scheduler.config.num_train_timesteps
        if alpha_bar is None:
            alpha_bar = float(scheduler.alphas_cumprod[timestep])
        stats: dict[str, float] = {"timestep": timestep, "alpha_bar": alpha_bar}
        if guidance_weight is not None:
            stats["guidance_weight"] = guidance_weight
        if err is not None:
            stats["err_norm"] = float(err.norm())
        if correction is not None:
            stats["grad_norm"] = float(correction.norm())
        if delta_out is not None:
            stats["delta_out_norm"] = float(delta_out.norm())
        if delta_x0 is not None:
            stats["delta_x0_norm"] = float(delta_x0.norm())
            if err is not None and float(err.pow(2).sum()) > 0:
                # Fraction of the (weighted) prefix error removed from x0_hat by this step's correction,
                # along the error direction: 1 = x0_hat moved exactly onto the target.
                stats["corr_frac"] = float((delta_x0 * err).sum() / err.pow(2).sum())
        self.step_stats.append(stats)
        metadata = {k: v for k, v in stats.items() if k != "guidance_weight"}
        self.processor.track(
            time=timestep / num_train,
            x_t=x_t,
            v_t=v_t,
            x1_t=x1_t,
            correction=correction,
            err=err,
            weights=weights,
            guidance_weight=guidance_weight,
            inference_delay=self.inference_delay,
            execution_horizon=self.overlap_len,
            mode=self.mode,
            weight_clip=self.weight_clip,
            jacobian=self.jacobian,
            **metadata,
        )


# ---------------------------------------------------------------------------------------------
# Policy-level helpers (used by DiffusionPolicy)
# ---------------------------------------------------------------------------------------------


def rtc_enabled(config: Any) -> bool:
    return getattr(config, "rtc_config", None) is not None and config.rtc_config.enabled


# `rtc_guidance_r2="posterior"`: the s2-estimation chunks use their own generator, seeded with
# R2_SAMPLES_SEED + call index, so they do not consume the global RNG stream (the guided chunk's noise is
# the same as for every other RTC variant). The floor keeps 1 / r2 finite for (near-)deterministic dims.
R2_SAMPLES_SEED = 2506_07339
R2_VAR_FLOOR = 1e-6


def estimate_prior_variance(
    model: DiffusionModel,
    global_cond: Tensor | None,
    batch_size: int,
    num_samples: int,
    generator: torch.Generator | None = None,
    var_floor: float = R2_VAR_FLOOR,
) -> Tensor:
    """Per-element conditional variance `s2` (B, horizon, A) of the unguided sampler's actions.

    Draws `num_samples` chunks per batch element with the original sampler (no RTC) from the same
    conditioning, i.e. the same observation snapshot, and returns their unbiased variance (normalized
    action space), clamped below at `var_floor`. Diagonal only: correlations across time / action dims
    are ignored (assumption of `rtc_guidance_r2="posterior"`).
    """
    cond = None if global_cond is None else global_cond.repeat_interleave(num_samples, dim=0)
    with torch.no_grad():
        samples = model.conditional_sample(batch_size * num_samples, global_cond=cond, generator=generator)
    samples = samples.view(batch_size, num_samples, *samples.shape[1:])
    return samples.var(dim=1, unbiased=True).clamp_min(var_floor)


def make_rtc_processor(policy: DiffusionPolicy) -> RTCProcessor | None:
    """Build the RTC processor from `policy.config.rtc_config` and warn about costly settings."""
    config = policy.config
    if config.rtc_config is None:
        return None
    processor = RTCProcessor(config.rtc_config)
    vjp_guidance = config.rtc_guidance_mode == "guidance" and config.rtc_guidance_jacobian == "vjp"
    if config.rtc_config.enabled:
        num_steps = policy.diffusion.num_inference_steps
        if vjp_guidance and num_steps > MANY_GUIDED_STEPS_WARNING_THRESHOLD:
            logger.warning(
                "RTC guidance with %s and %d inference steps: every step adds a VJP through the U-Net, "
                "so latency grows a lot (and early DDPM steps have alpha_bar ~ 0). Consider "
                "--policy.noise_scheduler_type=DDIM --policy.num_inference_steps=10 (no retraining "
                "needed, see `set_inference_noise_scheduler`).",
                config.noise_scheduler_type,
                num_steps,
            )
        if config.compile_model and vjp_guidance:
            logger.warning(
                "RTC guidance with compile_model=True is untested (autograd through a compiled U-Net)."
            )
    # Restart the s2-estimation noise stream (rtc_guidance_r2="posterior").
    policy._rtc_r2_calls = 0
    return processor


def set_inference_noise_scheduler(
    policy: DiffusionPolicy, scheduler_type: str, num_inference_steps: int | None = None
) -> None:
    """Swap the sampler at inference time (e.g. a DDPM-trained checkpoint sampled with DDIM).

    The noise scheduler holds no learned state; the beta schedule and prediction type come from the
    (unchanged) training config. Equivalent to loading with `--policy.noise_scheduler_type=DDIM
    --policy.num_inference_steps=N`.
    """
    from .modeling_diffusion import _make_noise_scheduler

    config = policy.config
    config.noise_scheduler_type = scheduler_type
    config.num_inference_steps = num_inference_steps
    model = policy.diffusion
    model.noise_scheduler = _make_noise_scheduler(
        scheduler_type,
        num_train_timesteps=config.num_train_timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        beta_schedule=config.beta_schedule,
        clip_sample=config.clip_sample,
        clip_sample_range=config.clip_sample_range,
        prediction_type=config.prediction_type,
    )
    model.num_inference_steps = (
        model.noise_scheduler.config.num_train_timesteps
        if num_inference_steps is None
        else num_inference_steps
    )


def push_observation_to_queues(policy: DiffusionPolicy, batch: dict[str, Tensor]) -> None:
    """Push one control-step observation into the policy's observation queues (no inference).

    Mirrors the preprocessing in `DiffusionPolicy.select_action` (camera stacking, queue filling on
    the first call). The caller's dict is not modified.
    """
    batch = dict(batch)
    batch.pop(ACTION, None)
    if policy.config.image_features:
        batch[OBS_IMAGES] = torch.stack([batch[key] for key in policy.config.image_features], dim=-4)
    with _OBS_QUEUE_LOCK:
        policy._queues = populate_queues(policy._queues, batch)


def snapshot_observations(policy: DiffusionPolicy) -> dict[str, Tensor]:
    """Stack the current observation queues into a model batch `(B, n_obs_steps, ...)`."""
    with _OBS_QUEUE_LOCK:
        queues = {k: list(q) for k, q in policy._queues.items() if k != ACTION}
    missing = [k for k, q in queues.items() if len(q) == 0]
    if missing:
        raise RuntimeError(
            f"Observation queues {missing} are empty. Call `predict_action_chunk(batch)` with the current "
            "observation (push_observation=True) or `policy.observe(batch)` first."
        )
    return {k: torch.stack(q, dim=1) for k, q in queues.items()}


def _as_batched_actions(x: Tensor, batch_size: int, action_dim: int, name: str) -> Tensor:
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim != 3:
        raise ValueError(f"`{name}` must have shape (T, A) or (B, T, A), got {tuple(x.shape)}")
    if x.shape[-1] != action_dim:
        raise ValueError(f"`{name}` action dim {x.shape[-1]} != policy action dim {action_dim}")
    if x.shape[0] == 1 and batch_size > 1:
        x = x.expand(batch_size, -1, -1)
    if x.shape[0] != batch_size:
        raise ValueError(f"`{name}` batch size {x.shape[0]} != observation batch size {batch_size}")
    return x


def build_guide(
    policy: DiffusionPolicy,
    *,
    batch_size: int,
    inference_delay: int | None,
    prev_chunk_left_over: Tensor | None,
    execution_horizon: int | None = None,
    prev_executed_actions: Tensor | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> DiffusionRTCGuide:
    """Build the per-call guide (targets/weights aligned to the raw horizon)."""
    config = policy.config
    rtc_cfg = config.rtc_config
    horizon, n_obs_steps = config.horizon, config.n_obs_steps
    action_dim = config.action_feature.shape[0]
    h_eff = executable_horizon(horizon, n_obs_steps)

    leftover = None
    if prev_chunk_left_over is not None and prev_chunk_left_over.numel() > 0:
        leftover = _as_batched_actions(prev_chunk_left_over, batch_size, action_dim, "prev_chunk_left_over")

    eh = rtc_cfg.execution_horizon if execution_horizon is None else execution_horizon
    overlap_len, delay = resolve_overlap(
        eh, 0 if leftover is None else leftover.shape[1], h_eff, inference_delay
    )

    processor = policy.rtc_processor if policy.rtc_processor is not None else RTCProcessor(rtc_cfg)
    exec_weights = processor.get_prefix_weights(delay, overlap_len, h_eff).to(torch.float32)

    past = None
    if config.rtc_constrain_past_actions and prev_executed_actions is not None:
        past = _as_batched_actions(prev_executed_actions, batch_size, action_dim, "prev_executed_actions")

    target, weights = build_raw_prefix(
        exec_weights,
        leftover,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        batch_size=batch_size,
        action_dim=action_dim,
        past_actions=past,
        dtype=dtype,
        device=device,
    )
    return DiffusionRTCGuide(
        mode=config.rtc_guidance_mode,
        target=target,
        weights=weights,
        max_guidance_weight=rtc_cfg.max_guidance_weight,
        prediction_type=config.prediction_type,
        inference_delay=delay,
        overlap_len=overlap_len,
        processor=processor,
        weight_clip=config.rtc_guidance_weight_clip,
        jacobian=config.rtc_guidance_jacobian,
        r2_clip=config.rtc_r2_clip,
    )


def rtc_predict_action_chunk(
    policy: DiffusionPolicy,
    batch: dict[str, Tensor],
    noise: Tensor | None = None,
    inference_delay: int | None = None,
    prev_chunk_left_over: Tensor | None = None,
    execution_horizon: int | None = None,
    prev_executed_actions: Tensor | None = None,
    push_observation: bool = True,
    generator: torch.Generator | None = None,
    **unused_kwargs: Any,
) -> Tensor:
    """RTC version of `DiffusionPolicy.predict_action_chunk`.

    Args:
        batch: the *current* observation (pushed into the queues if `push_observation`). Every control
            step must push exactly one observation: through this call at steps where inference starts,
            and through `policy.observe(batch)` at all other steps.
        inference_delay: number of steps executed from `prev_chunk_left_over` before the new chunk
            takes over (frozen prefix length).
        prev_chunk_left_over: (T, A) or (B, T, A) normalized actions of the previous chunk, index 0 being
            the action for the time of the current observation.
        execution_horizon: overrides `rtc_config.execution_horizon` (overlap length).
        prev_executed_actions: (K, A) or (B, K, A) normalized actions actually executed up to time
            `t - 1` (only used with `rtc_constrain_past_actions`).
        push_observation: set False if the current observation was already pushed with `observe`.

    Returns:
        (B, H_eff, A) normalized actions for times `t, ..., t + H_eff - 1`.
    """
    if unused_kwargs:
        logger.debug("Ignoring unused RTC kwargs: %s", sorted(unused_kwargs))
    config = policy.config
    if push_observation:
        push_observation_to_queues(policy, batch)
    obs = snapshot_observations(policy)

    model = policy.diffusion
    batch_size, n_obs_steps = obs[OBS_STATE].shape[:2]
    assert n_obs_steps == config.n_obs_steps

    global_cond = model._prepare_global_conditioning(obs)
    guide = build_guide(
        policy,
        batch_size=batch_size,
        inference_delay=inference_delay,
        prev_chunk_left_over=prev_chunk_left_over,
        execution_horizon=execution_horizon,
        prev_executed_actions=prev_executed_actions,
        device=global_cond.device,
        dtype=global_cond.dtype,
    )
    if config.rtc_guidance_r2 == "posterior" and guide.active and guide.mode == "guidance":
        calls = getattr(policy, "_rtc_r2_calls", 0)
        policy._rtc_r2_calls = calls + 1
        r2_gen = torch.Generator(device=global_cond.device).manual_seed(R2_SAMPLES_SEED + calls)
        guide.prior_var = estimate_prior_variance(
            model, global_cond, batch_size, config.rtc_r2_num_samples, generator=r2_gen
        ).to(dtype=guide.target.dtype)
    actions = model.conditional_sample(
        batch_size, global_cond=global_cond, generator=generator, noise=noise, rtc_guide=guide
    )
    policy.last_rtc_info = {
        "mode": guide.mode,
        "active": guide.active,
        "inference_delay": guide.inference_delay,
        "overlap_len": guide.overlap_len,
        "leftover_len": 0 if prev_chunk_left_over is None else int(prev_chunk_left_over.shape[-2]),
        "weights": guide.weights.detach().cpu(),
        "steps": guide.step_stats,
        "prior_var": None if guide.prior_var is None else guide.prior_var.detach().cpu(),
    }
    return actions[:, exec_to_raw_index(0, config.n_obs_steps) :]
