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

"""Tests for Real-Time Chunking (RTC) in DiffusionPolicy (`policies/diffusion/rtc_diffusion.py`)."""

import json
import logging

import pytest
import torch
from torch import nn

from lerobot.configs import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature, RTCAttentionSchedule
from lerobot.policies.rtc import RTCConfig, RTCProcessor
from lerobot.utils.constants import OBS_STATE

pytest.importorskip("diffusers")

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig  # noqa: E402
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy  # noqa: E402
from lerobot.policies.diffusion.rtc_diffusion import (  # noqa: E402
    DiffusionRTCGuide,
    apply_guidance_correction,
    build_guide,
    build_raw_prefix,
    exec_to_raw_index,
    executable_horizon,
    flow_scale,
    guidance_weight,
    posterior_guidance_weights,
    posterior_r2,
    resolve_overlap,
    set_inference_noise_scheduler,
    snapshot_observations,
    tweedie_x0,
)
from lerobot.policies.utils import populate_queues  # noqa: E402

STATE_DIM, ENV_DIM, ACTION_DIM = 3, 4, 2
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


# ------------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------------


def make_config(rtc_config=None, *, images=False, **overrides) -> DiffusionConfig:
    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(ENV_DIM,)),
    }
    if images:
        input_features["observation.images.top"] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32))
    kwargs = {
        "input_features": input_features,
        "output_features": {"action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
        "n_obs_steps": 2,
        "horizon": 16,
        "n_action_steps": 8,
        "down_dims": (16, 32),
        "diffusion_step_embed_dim": 16,
        "noise_scheduler_type": "DDIM",
        "num_inference_steps": 10,
        "pretrained_backbone_weights": None,
        "use_group_norm": True,
        "spatial_softmax_num_keypoints": 4,
        "rtc_config": rtc_config,
        "device": "cpu",
    }
    kwargs.update(overrides)
    return DiffusionConfig(**kwargs)


def make_policy(rtc_config=None, seed: int = 0, device: str = "cpu", **overrides) -> DiffusionPolicy:
    torch.manual_seed(seed)
    return DiffusionPolicy(make_config(rtc_config, **overrides)).to(device).eval()


def rtc_cfg(schedule=RTCAttentionSchedule.ZEROS, beta=10.0, execution_horizon=10, **kw) -> RTCConfig:
    return RTCConfig(
        enabled=True,
        prefix_attention_schedule=schedule,
        max_guidance_weight=beta,
        execution_horizon=execution_horizon,
        **kw,
    )


def make_obs(batch_size: int = 1, seed: int = 0, device: str = "cpu", images: bool = False) -> dict:
    g = torch.Generator().manual_seed(seed)
    obs = {
        OBS_STATE: torch.randn(batch_size, STATE_DIM, generator=g),
        "observation.environment_state": torch.randn(batch_size, ENV_DIM, generator=g),
    }
    if images:
        obs["observation.images.top"] = torch.rand(batch_size, 3, 32, 32, generator=g)
    return {k: v.to(device) for k, v in obs.items()}


def make_noise(policy: DiffusionPolicy, batch_size: int, seed: int = 1, device: str = "cpu") -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    shape = (batch_size, policy.config.horizon, ACTION_DIM)
    return torch.randn(shape, generator=g).to(device=device, dtype=next(policy.parameters()).dtype)


def reference_generate_actions(model, batch: dict, noise: torch.Tensor) -> torch.Tensor:
    """Verbatim copy of the pre-RTC `generate_actions` / `conditional_sample` (n_action_steps slice)."""
    n_obs_steps = batch[OBS_STATE].shape[1]
    global_cond = model._prepare_global_conditioning(batch)
    sample = noise
    model.noise_scheduler.set_timesteps(model.num_inference_steps)
    for t in model.noise_scheduler.timesteps:
        model_output = model.unet(
            sample,
            torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
            global_cond=global_cond,
        )
        sample = model.noise_scheduler.step(model_output, t, sample, generator=None).prev_sample
    start = n_obs_steps - 1
    return sample[:, start : start + model.config.n_action_steps]


class GaussianPriorDenoiser(nn.Module):
    """Exact MMSE denoiser for x0 ~ N(0, var * K) with an RBF kernel K over time (dims independent).

    `E[x0 | x_t] = sqrt(a) K (a K + (1 - a) I)^-1 x_t`. Unlike a randomly initialised U-Net, this is
    a real denoiser, so the quality of PiGDM guidance can be tested meaningfully.
    """

    def __init__(self, scheduler, horizon: int, prediction_type: str, var: float = 0.25, lengthscale=3.0):
        super().__init__()
        t = torch.arange(horizon, dtype=torch.float64)
        k = var * torch.exp(-0.5 * (t[:, None] - t[None]) ** 2 / lengthscale**2)
        evals, evecs = torch.linalg.eigh(k + 1e-6 * torch.eye(horizon, dtype=torch.float64))
        self.register_buffer("evals", evals)
        self.register_buffer("evecs", evecs)
        self.dummy = nn.Parameter(torch.zeros(()))  # keeps `get_device_from_parameters` working
        self.scheduler = scheduler
        self.prediction_type = prediction_type

    def x0(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        a = self.scheduler.alphas_cumprod.to(torch.float64)[timestep.cpu()].to(x)[:, None, None]
        lam = self.evals.to(x)[None, :, None]
        u = self.evecs.to(x)
        coef = a.sqrt() * lam / (a * lam + (1 - a))
        z = torch.einsum("st,bsa->bta", u, x)
        return torch.einsum("st,bta->bsa", u, coef * z) + 0 * self.dummy

    def forward(self, x, timestep, global_cond=None):
        x0 = self.x0(x, timestep)
        if self.prediction_type == "sample":
            return x0
        a = self.scheduler.alphas_cumprod.to(torch.float64)[timestep.cpu()].to(x)[:, None, None]
        return (x - a.sqrt() * x0) / (1 - a).sqrt()

    def sample_prior(self, batch_size: int, action_dim: int, generator: torch.Generator) -> torch.Tensor:
        z = torch.randn(batch_size, self.evals.shape[0], action_dim, generator=generator, dtype=torch.float64)
        return torch.einsum("st,bta->bsa", self.evecs, self.evals.clamp_min(0).sqrt()[None, :, None] * z)


class EpsilonFromSampleUNet(nn.Module):
    """Turns an x0-predicting network into the equivalent epsilon-predicting network."""

    def __init__(self, sample_unet: nn.Module, scheduler):
        super().__init__()
        self.inner = sample_unet
        self.scheduler = scheduler

    def forward(self, x, timestep, global_cond=None):
        a = self.scheduler.alphas_cumprod.to(x.dtype)[timestep.cpu()].to(x.device)[:, None, None]
        return (x - a.sqrt() * self.inner(x, timestep, global_cond=global_cond)) / (1 - a).sqrt()


def install_gaussian_denoiser(policy: DiffusionPolicy) -> GaussianPriorDenoiser:
    model = policy.diffusion
    den = GaussianPriorDenoiser(model.noise_scheduler, policy.config.horizon, policy.config.prediction_type)
    den = den.to(next(policy.parameters()).device)
    model.unet = den
    return den


def prefix_error(actions: torch.Tensor, leftover: torch.Tensor, weights_exec: torch.Tensor) -> float:
    n = leftover.shape[-2]
    return float(((actions[:, :n] - leftover) * weights_exec[:n, None]).norm())


# ------------------------------------------------------------------------------------------------
# 1. RTC off / leftover None == original implementation (bit-identical)
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("prediction_type", ["epsilon", "sample"])
@pytest.mark.parametrize("scheduler", ["DDIM", "DDPM"])
@pytest.mark.parametrize("rtc", [None, "disabled"])
def test_rtc_off_is_bit_identical_to_reference(prediction_type, scheduler, rtc):
    rtc_config = None if rtc is None else RTCConfig(enabled=False)
    policy = make_policy(rtc_config, prediction_type=prediction_type, noise_scheduler_type=scheduler)
    obs = make_obs(batch_size=2)
    noise = make_noise(policy, 2)

    policy._queues = populate_queues(policy._queues, dict(obs))
    torch.manual_seed(123)  # DDPM draws variance noise from the global RNG
    actions = policy.predict_action_chunk(dict(obs), noise=noise.clone())

    stacked = {k: torch.stack([v, v], dim=1) for k, v in obs.items()}
    torch.manual_seed(123)
    with torch.no_grad():
        expected = reference_generate_actions(policy.diffusion, stacked, noise.clone())
    assert torch.equal(actions, expected)


@pytest.mark.parametrize("prediction_type", ["epsilon", "sample"])
@pytest.mark.parametrize("scheduler", ["DDIM", "DDPM"])
@pytest.mark.parametrize("images", [False, True])
@pytest.mark.parametrize("mode", ["guidance", "replacement", "none"])
def test_rtc_enabled_without_leftover_is_bit_identical(prediction_type, scheduler, images, mode):
    off = make_policy(None, prediction_type=prediction_type, noise_scheduler_type=scheduler, images=images)
    on = make_policy(
        rtc_cfg(RTCAttentionSchedule.EXP),
        prediction_type=prediction_type,
        noise_scheduler_type=scheduler,
        images=images,
        rtc_guidance_mode=mode,
    )
    obs = make_obs(batch_size=2, images=images)
    noise = make_noise(off, 2)

    torch.manual_seed(7)
    a_off = off.select_action(dict(obs), noise=noise.clone())  # populates queues, returns step 0
    off_chunk = torch.stack([a_off] + [off.select_action(dict(obs)) for _ in range(7)], dim=1)

    torch.manual_seed(7)
    a_on = on.predict_action_chunk(
        dict(obs), noise=noise.clone(), inference_delay=3, prev_chunk_left_over=None
    )

    h_eff = executable_horizon(on.config.horizon, on.config.n_obs_steps)
    assert a_on.shape == (2, h_eff, ACTION_DIM)
    assert torch.equal(a_on[:, : on.config.n_action_steps], off_chunk)
    assert on.last_rtc_info["active"] is False


def test_empty_leftover_and_zeros_schedule_with_zero_delay_are_unguided():
    on = make_policy(rtc_cfg(RTCAttentionSchedule.ZEROS))
    obs, noise = make_obs(2), make_noise(on, 2)
    base = on.predict_action_chunk(dict(obs), noise=noise.clone())
    on.reset()
    leftover = torch.rand(10, ACTION_DIM)
    zero_delay = on.predict_action_chunk(
        dict(obs), noise=noise.clone(), inference_delay=0, prev_chunk_left_over=leftover
    )
    on.reset()
    empty = on.predict_action_chunk(
        dict(obs), noise=noise.clone(), inference_delay=3, prev_chunk_left_over=torch.zeros(0, ACTION_DIM)
    )
    assert torch.equal(base, zero_delay)
    assert torch.equal(base, empty)


# ------------------------------------------------------------------------------------------------
# 2. All-zero weights through the *guided* code path == unguided
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("prediction_type", ["epsilon", "sample"])
def test_zero_weights_guided_path_matches_unguided(prediction_type, monkeypatch):
    policy = make_policy(rtc_cfg(), prediction_type=prediction_type)
    model = policy.diffusion
    obs = make_obs(2)
    stacked = {k: torch.stack([v, v], dim=1) for k, v in obs.items()}
    global_cond = model._prepare_global_conditioning(stacked)
    noise = make_noise(policy, 2)
    horizon = policy.config.horizon

    def guide():
        return DiffusionRTCGuide(
            mode="guidance",
            target=torch.randn(2, horizon, ACTION_DIM),
            weights=torch.zeros(horizon),
            max_guidance_weight=10.0,
            prediction_type=prediction_type,
        )

    with torch.no_grad():
        unguided = model.conditional_sample(
            2, global_cond=global_cond, noise=noise.clone(), rtc_guide=guide()
        )
        monkeypatch.setattr(DiffusionRTCGuide, "active", property(lambda self: True))
        forced = model.conditional_sample(2, global_cond=global_cond, noise=noise.clone(), rtc_guide=guide())
    torch.testing.assert_close(forced, unguided, rtol=1e-5, atol=1e-6)


# ------------------------------------------------------------------------------------------------
# 3. epsilon and sample corrections are equivalent
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("alpha_bar", [2.4e-7, 0.0195, 0.478, 0.967, 0.9994])
@pytest.mark.parametrize("weight", [0.3, 1.0, 10.0])
def test_epsilon_and_sample_corrections_imply_same_x0(alpha_bar, weight):
    g = torch.Generator().manual_seed(0)
    x_t, eps, grad = (torch.randn(3, 16, 2, generator=g, dtype=torch.float64) for _ in range(3))
    x0 = tweedie_x0(eps, x_t, alpha_bar, "epsilon")
    eps_new = apply_guidance_correction(eps, grad, alpha_bar, weight, "epsilon")
    x0_new_from_eps = tweedie_x0(eps_new, x_t, alpha_bar, "epsilon")
    x0_new_from_sample = apply_guidance_correction(x0, grad, alpha_bar, weight, "sample")
    torch.testing.assert_close(x0_new_from_eps, x0_new_from_sample, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("scheduler", ["DDIM", "DDPM"])
@pytest.mark.parametrize("clip_sample", [True, False])
@pytest.mark.parametrize("weight_clip", ["pigdm", "flow"])
def test_epsilon_and_sample_guided_sampling_are_equivalent(scheduler, clip_sample, weight_clip):
    """Same x0-network under both parameterizations -> same guided trajectory.

    Runs in float32: the U-Net's sinusoidal timestep embedding does not support float64.
    """
    kw = {
        "noise_scheduler_type": scheduler,
        "num_inference_steps": 10,
        "clip_sample": clip_sample,
        "rtc_guidance_weight_clip": weight_clip,
    }
    p_sample = make_policy(rtc_cfg(RTCAttentionSchedule.EXP), prediction_type="sample", **kw)
    p_eps = make_policy(rtc_cfg(RTCAttentionSchedule.EXP), prediction_type="epsilon", **kw)
    p_eps.load_state_dict(p_sample.state_dict())
    p_eps.diffusion.unet = EpsilonFromSampleUNet(p_eps.diffusion.unet, p_eps.diffusion.noise_scheduler)

    obs = make_obs(3)
    noise = make_noise(p_sample, 3)
    leftover = torch.rand(3, 12, ACTION_DIM, generator=torch.Generator().manual_seed(5)) * 2 - 1
    out = []
    for p in (p_sample, p_eps):
        torch.manual_seed(0)
        out.append(
            p.predict_action_chunk(
                dict(obs), noise=noise.clone(), inference_delay=3, prev_chunk_left_over=leftover
            )
        )
    assert p_sample.last_rtc_info["active"] and p_eps.last_rtc_info["active"]
    torch.testing.assert_close(out[0], out[1], rtol=1e-4, atol=2e-5)


# ------------------------------------------------------------------------------------------------
# 4. Guidance reduces the prefix error (exact Gaussian-prior denoiser)
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("prediction_type", ["epsilon", "sample"])
@pytest.mark.parametrize(
    "schedule", [RTCAttentionSchedule.ZEROS, RTCAttentionSchedule.EXP, RTCAttentionSchedule.LINEAR]
)
@pytest.mark.parametrize("weight_clip", ["pigdm", "flow"])
@pytest.mark.parametrize("jacobian", ["vjp", "identity"])
def test_guidance_reduces_prefix_error(prediction_type, schedule, weight_clip, jacobian):
    batch_size, delay = 16, 4
    guided = make_policy(
        rtc_cfg(schedule, beta=100.0),
        prediction_type=prediction_type,
        rtc_guidance_weight_clip=weight_clip,
        rtc_guidance_jacobian=jacobian,
    )
    naive = make_policy(
        rtc_cfg(schedule, beta=100.0), prediction_type=prediction_type, rtc_guidance_mode="none"
    )
    den = install_gaussian_denoiser(guided)
    install_gaussian_denoiser(naive)

    # The leftover is an in-distribution trajectory: the raw slots of another prior sample, kept inside
    # the clip_sample range (targets outside [-1, 1] are unreachable by construction).
    raw = (
        den.sample_prior(batch_size, ACTION_DIM, torch.Generator().manual_seed(3)).float().clamp(-0.95, 0.95)
    )
    leftover = raw[:, exec_to_raw_index(0, 2) :][:, :12]
    obs, noise = make_obs(batch_size), make_noise(guided, batch_size)

    a_naive = naive.predict_action_chunk(
        dict(obs), noise=noise.clone(), inference_delay=delay, prev_chunk_left_over=leftover
    )
    a_guided = guided.predict_action_chunk(
        dict(obs), noise=noise.clone(), inference_delay=delay, prev_chunk_left_over=leftover
    )
    w_exec = guided.last_rtc_info["weights"][exec_to_raw_index(0, 2) :]
    assert guided.last_rtc_info["active"]

    err_naive = prefix_error(a_naive, leftover, w_exec)
    err_guided = prefix_error(a_guided, leftover, w_exec)
    assert err_guided < 0.3 * err_naive, (err_guided, err_naive)
    # The frozen part (first `delay` steps, W == 1) is matched closely. PiGDM is a soft constraint
    # (and r2 = 1 - a_t assumes a unit-variance prior), so it is not an exact projection.
    frozen_rms = float((a_guided[:, :delay] - leftover[:, :delay]).pow(2).mean().sqrt())
    naive_rms = float((a_naive[:, :delay] - leftover[:, :delay]).pow(2).mean().sqrt())
    assert frozen_rms < 0.2 and frozen_rms < 0.3 * naive_rms, (frozen_rms, naive_rms)


def test_larger_beta_does_not_increase_prefix_error():
    errs = []
    for beta in (1.0, 10.0, 100.0, 1e4):
        policy = make_policy(rtc_cfg(RTCAttentionSchedule.ZEROS, beta=beta))
        den = install_gaussian_denoiser(policy)
        raw = den.sample_prior(8, ACTION_DIM, torch.Generator().manual_seed(3)).float().clamp(-0.95, 0.95)
        leftover = raw[:, 1:13]
        a = policy.predict_action_chunk(
            make_obs(8), noise=make_noise(policy, 8), inference_delay=5, prev_chunk_left_over=leftover
        )
        errs.append(float((a[:, :5] - leftover[:, :5]).norm()))
    assert errs[0] >= errs[1] >= errs[2], errs


def test_replacement_hits_target_exactly_on_frozen_prefix():
    delay = 4
    policy = make_policy(rtc_cfg(RTCAttentionSchedule.EXP), rtc_guidance_mode="replacement")
    leftover = torch.rand(2, 12, ACTION_DIM) * 2 - 1
    a = policy.predict_action_chunk(
        make_obs(2), noise=make_noise(policy, 2), inference_delay=delay, prev_chunk_left_over=leftover
    )
    weights = policy.last_rtc_info["weights"]
    hard = (weights >= 1)[exec_to_raw_index(0, 2) :]
    assert hard[:delay].all() and not hard[delay:].any()  # EXP: only the frozen prefix has W == 1
    assert torch.equal(a[:, :delay], leftover[:, :delay])


# ------------------------------------------------------------------------------------------------
# 5. Index mapping: leftover[i] <-> raw (n_obs_steps - 1 + i)
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n_obs_steps", [1, 2, 3])
@pytest.mark.parametrize("leftover_len", [3, 8, 30])
def test_index_mapping(n_obs_steps, leftover_len):
    horizon = 16
    h_eff = executable_horizon(horizon, n_obs_steps)
    assert h_eff == horizon - n_obs_steps + 1
    assert [exec_to_raw_index(i, n_obs_steps) for i in range(3)] == [n_obs_steps - 1 + i for i in range(3)]

    leftover = torch.arange(2 * leftover_len * 2, dtype=torch.float32).reshape(2, leftover_len, 2) + 1
    exec_weights = torch.linspace(1, 0.1, h_eff)
    target, weights = build_raw_prefix(
        exec_weights, leftover, horizon=horizon, n_obs_steps=n_obs_steps, batch_size=2, action_dim=2
    )
    n = min(leftover_len, h_eff)
    for i in range(n):
        j = exec_to_raw_index(i, n_obs_steps)
        assert torch.equal(target[:, j], leftover[:, i])
        assert weights[j] == exec_weights[i]
    # Past slots and slots beyond the leftover are unconstrained.
    assert (weights[: n_obs_steps - 1] == 0).all() and (target[:, : n_obs_steps - 1] == 0).all()
    assert (weights[exec_to_raw_index(n, n_obs_steps) :] == 0).all()


def test_past_actions_placement():
    exec_weights = torch.zeros(14)
    past = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])  # last row = time t - 1
    target, weights = build_raw_prefix(
        exec_weights, None, horizon=16, n_obs_steps=3, batch_size=1, action_dim=2, past_actions=past
    )
    assert torch.equal(target[0, :2], past[0, -2:])  # raw 0 = t - 2, raw 1 = t - 1
    assert weights.tolist()[:3] == [1.0, 1.0, 0.0]


@pytest.mark.parametrize(
    ("eh", "left", "h_eff", "delay", "expected"),
    [
        (10, 12, 15, 3, (10, 3)),
        (10, 5, 15, 7, (5, 5)),
        (20, 30, 15, 2, (15, 2)),
        (10, 0, 15, 3, (0, 0)),
        (10, 12, 15, None, (10, 0)),
        (10, 12, 15, -2, (10, 0)),
    ],
)
def test_resolve_overlap(eh, left, h_eff, delay, expected):
    assert resolve_overlap(eh, left, h_eff, delay) == expected


@pytest.mark.parametrize("schedule", list(RTCAttentionSchedule))
@pytest.mark.parametrize("delay", [0, 2, 5, 12])
def test_prefix_weights_match_rtc_processor(schedule, delay):
    cfg = rtc_cfg(schedule, execution_horizon=10)
    policy = make_policy(cfg)
    leftover = torch.rand(12, ACTION_DIM)
    policy.predict_action_chunk(make_obs(1), inference_delay=delay, prev_chunk_left_over=leftover)
    weights = policy.last_rtc_info["weights"]
    overlap, d = resolve_overlap(10, 12, 15, delay)
    expected = RTCProcessor(cfg).get_prefix_weights(d, overlap, 15)
    assert weights[0] == 0
    torch.testing.assert_close(weights[1:], expected)


def test_constrain_past_actions_sets_weights():
    policy = make_policy(rtc_cfg(), rtc_constrain_past_actions=True)
    past = torch.full((1, ACTION_DIM), 0.5)
    policy.predict_action_chunk(
        make_obs(1), inference_delay=2, prev_chunk_left_over=torch.rand(10, 2), prev_executed_actions=past
    )
    assert policy.last_rtc_info["weights"][0] == 1.0
    policy.config.rtc_constrain_past_actions = False
    policy.predict_action_chunk(
        make_obs(1), inference_delay=2, prev_chunk_left_over=torch.rand(10, 2), prev_executed_actions=past
    )
    assert policy.last_rtc_info["weights"][0] == 0.0


# ------------------------------------------------------------------------------------------------
# 6. No parameter gradients
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("prediction_type", ["epsilon", "sample"])
@pytest.mark.parametrize("jacobian", ["vjp", "identity"])
def test_no_param_grads_after_guided_inference(prediction_type, jacobian):
    policy = make_policy(
        rtc_cfg(RTCAttentionSchedule.EXP),
        prediction_type=prediction_type,
        images=True,
        rtc_guidance_jacobian=jacobian,
    )
    requires_grad_before = [p.requires_grad for p in policy.parameters()]
    policy.predict_action_chunk(
        make_obs(2, images=True), inference_delay=3, prev_chunk_left_over=torch.rand(10, 2)
    )
    assert policy.last_rtc_info["active"]
    assert all(p.grad is None for p in policy.parameters())
    assert [p.requires_grad for p in policy.parameters()] == requires_grad_before


@pytest.mark.parametrize("jacobian", ["vjp", "identity"])
def test_guided_inference_works_under_inference_mode(jacobian):
    policy = make_policy(rtc_cfg(RTCAttentionSchedule.EXP), rtc_guidance_jacobian=jacobian)
    obs, noise, leftover = make_obs(2), make_noise(policy, 2), torch.rand(10, 2)
    expected = policy.predict_action_chunk(
        dict(obs), noise=noise.clone(), inference_delay=3, prev_chunk_left_over=leftover
    )
    policy.reset()
    with torch.inference_mode():
        actual = policy.predict_action_chunk(
            dict(obs), noise=noise.clone(), inference_delay=3, prev_chunk_left_over=leftover
        )
    torch.testing.assert_close(actual, expected)


# ------------------------------------------------------------------------------------------------
# 7. Shapes / dtypes / devices / batch
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("leftover_batched", [False, True])
@pytest.mark.parametrize(
    ("mode", "jacobian"),
    [("guidance", "vjp"), ("guidance", "identity"), ("replacement", "vjp"), ("none", "vjp")],
)
def test_shapes_dtypes_devices(device, batch_size, leftover_batched, mode, jacobian):
    policy = make_policy(
        rtc_cfg(RTCAttentionSchedule.LINEAR),
        device=device,
        rtc_guidance_mode=mode,
        rtc_guidance_jacobian=jacobian,
    )
    leftover = torch.rand(batch_size, 10, ACTION_DIM) if leftover_batched else torch.rand(10, ACTION_DIM)
    actions = policy.predict_action_chunk(
        make_obs(batch_size, device=device), inference_delay=2, prev_chunk_left_over=leftover.to(device)
    )
    assert actions.shape == (batch_size, 15, ACTION_DIM)
    assert actions.dtype == torch.float32
    assert actions.device.type == device
    assert torch.isfinite(actions).all()


def test_batch_elements_are_independent():
    """Guided sampling of a batch == guided sampling of each element alone."""
    policy = make_policy(rtc_cfg(RTCAttentionSchedule.EXP))
    obs, noise = make_obs(3), make_noise(policy, 3)
    leftover = torch.rand(3, 10, ACTION_DIM)
    full = policy.predict_action_chunk(
        dict(obs), noise=noise.clone(), inference_delay=3, prev_chunk_left_over=leftover
    )
    for b in range(3):
        policy.reset()
        single = policy.predict_action_chunk(
            {k: v[b : b + 1] for k, v in obs.items()},
            noise=noise[b : b + 1].clone(),
            inference_delay=3,
            prev_chunk_left_over=leftover[b],
        )
        torch.testing.assert_close(single, full[b : b + 1], rtol=1e-4, atol=1e-5)


def test_bad_leftover_shapes_raise():
    policy = make_policy(rtc_cfg())
    with pytest.raises(ValueError, match="action dim"):
        policy.predict_action_chunk(make_obs(1), inference_delay=1, prev_chunk_left_over=torch.rand(10, 3))
    policy.reset()
    with pytest.raises(ValueError, match="batch size"):
        policy.predict_action_chunk(make_obs(2), inference_delay=1, prev_chunk_left_over=torch.rand(3, 10, 2))


# ------------------------------------------------------------------------------------------------
# 8. Config compatibility
# ------------------------------------------------------------------------------------------------


def test_old_config_json_without_rtc_fields_loads(tmp_path):
    cfg = make_config(None)
    cfg._save_pretrained(tmp_path)
    data = json.loads((tmp_path / "config.json").read_text())
    for key in (
        "rtc_config",
        "rtc_guidance_mode",
        "rtc_constrain_past_actions",
        "rtc_guidance_weight_clip",
        "rtc_guidance_jacobian",
    ):
        data.pop(key)
    (tmp_path / "config.json").write_text(json.dumps(data))

    loaded = PreTrainedConfig.from_pretrained(tmp_path)
    assert isinstance(loaded, DiffusionConfig)
    assert loaded.rtc_config is None
    assert loaded.rtc_guidance_mode == "guidance"
    assert loaded.rtc_constrain_past_actions is False
    assert loaded.rtc_guidance_jacobian == "vjp"


def test_config_roundtrip_with_rtc(tmp_path):
    cfg = make_config(rtc_cfg(RTCAttentionSchedule.EXP, beta=3.0), rtc_guidance_mode="replacement")
    cfg._save_pretrained(tmp_path)
    loaded = PreTrainedConfig.from_pretrained(tmp_path)
    assert loaded.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.EXP
    assert loaded.rtc_config.max_guidance_weight == 3.0
    assert loaded.rtc_guidance_mode == "replacement"


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="rtc_guidance_mode"):
        make_config(rtc_guidance_mode="inpaint")
    with pytest.raises(ValueError, match="rtc_guidance_weight_clip"):
        make_config(rtc_guidance_weight_clip="foo")
    with pytest.raises(ValueError, match="rtc_guidance_jacobian"):
        make_config(rtc_guidance_jacobian="foo")
    with pytest.raises(ValueError, match="rtc_guidance_r2"):
        make_config(rtc_guidance_r2="foo")
    with pytest.raises(ValueError, match="rtc_r2_clip"):
        make_config(rtc_r2_clip="foo")
    with pytest.raises(ValueError, match="rtc_r2_num_samples"):
        make_config(rtc_r2_num_samples=1)


def test_v_prediction_raises():
    with pytest.raises(ValueError, match="prediction_type"):
        tweedie_x0(torch.zeros(1), torch.zeros(1), 0.5, "v_prediction")
    with pytest.raises(ValueError, match="prediction_type"):
        DiffusionRTCGuide(
            mode="guidance",
            target=torch.zeros(1, 4, 2),
            weights=torch.ones(4),
            max_guidance_weight=1.0,
            prediction_type="v_prediction",
        )


# ------------------------------------------------------------------------------------------------
# Cross-check against LeRobot's flow-matching RTCProcessor
# ------------------------------------------------------------------------------------------------


def _vp_to_lerobot_flow(alpha_bar: float) -> tuple[float, float]:
    """LeRobot flow time (1 = noise, 0 = data) and scale s with x_flow = s * x_vp."""
    r = ((1 - alpha_bar) / alpha_bar) ** 0.5
    time = r / (1 + r)
    return time, ((1 - time) ** 2 + time**2) ** 0.5


def _diffusion_x0_delta(x0_of, x_vp, leftover, weights, alpha_bar, beta, clip):
    """Implied x0 update of this module's guidance for a given denoiser `x0_of` (VP variables)."""
    with torch.enable_grad():
        x = x_vp.clone().requires_grad_(True)
        x0 = x0_of(x)
        err = (leftover - x0.detach()) * weights[None, :, None]
        (grad,) = torch.autograd.grad(x0, x, grad_outputs=err)
    w = guidance_weight(alpha_bar, beta, clip)
    return apply_guidance_correction(x0.detach(), grad, alpha_bar, w, "sample") - x0.detach()


def _clip_regime(alpha_bar: float, beta: float) -> tuple[bool, bool]:
    flow_unclipped = 1 / (alpha_bar * (1 - alpha_bar)) ** 0.5 <= beta
    pigdm_unclipped = 1 / (1 - alpha_bar) <= beta
    return flow_unclipped, pigdm_unclipped


@pytest.mark.parametrize("alpha_bar", [0.0195, 0.478, 0.967, 0.9994])
@pytest.mark.parametrize("beta", [1.5, 10.0, 1e6])
@pytest.mark.parametrize("schedule", [RTCAttentionSchedule.EXP, RTCAttentionSchedule.ZEROS])
def test_flow_clip_matches_lerobot_rtc_processor(alpha_bar, beta, schedule):
    """Weight/clip mapping vs LeRobot's flow-matching `RTCProcessor.denoise_step`.

    NOTE: `RTCProcessor.denoise_step` evaluates the velocity *before* `x_t.requires_grad_(True)`, so its
    "VJP" is taken with an identity Jacobian (correction == err). To compare the weight mapping only,
    the denoiser here is chosen such that the flow Jacobian is exactly the identity
    (x0_hat = s * x_vp, i.e. x1_hat = x_flow). The full-Jacobian case is covered by
    `test_guidance_matches_flow_pigdm_reference_with_true_vjp`.
    """
    horizon, delay, eh = 16, 4, 10
    processor = RTCProcessor(rtc_cfg(schedule, beta=beta, execution_horizon=eh))
    g = torch.Generator().manual_seed(0)
    x_vp = torch.randn(1, horizon, 2, generator=g, dtype=torch.float64)
    leftover = torch.randn(1, horizon, 2, generator=g, dtype=torch.float64)
    time, s = _vp_to_lerobot_flow(alpha_bar)
    weights = processor.get_prefix_weights(delay, eh, horizon).to(torch.float64)

    deltas = {
        clip: _diffusion_x0_delta(lambda x: s * x, x_vp, leftover, weights, alpha_bar, beta, clip)
        for clip in ("pigdm", "flow")
    }

    def velocity(xf):  # x1_hat = xf - time * v = xf
        return torch.zeros_like(xf)

    x_flow = s * x_vp
    v_guided = processor.denoise_step(x_flow, leftover, delay, time, velocity, execution_horizon=eh)
    delta_flow = -time * v_guided  # x1 = x - time * v

    torch.testing.assert_close(deltas["flow"], delta_flow, rtol=2e-5, atol=1e-6)
    if all(_clip_regime(alpha_bar, beta)):
        torch.testing.assert_close(deltas["pigdm"], delta_flow, rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize("alpha_bar", [0.0195, 0.478, 0.967, 0.9994])
@pytest.mark.parametrize("beta", [1.5, 10.0, 1e6])
def test_guidance_matches_flow_pigdm_reference_with_true_vjp(alpha_bar, beta):
    """Full-Jacobian check against the paper's flow formula (Black et al. 2025, Eq. 2-4), using a
    dense nonlinear denoiser. Paper convention: tau 0 = noise, 1 = data, A_tau = tau A + (1 - tau) eps,
    v' = v + min(beta, (1 - tau) / (tau r^2)) * VJP(A1_hat wrt A_tau, e), A1_hat = A_tau + (1 - tau) v.
    """
    horizon = 16
    g = torch.Generator().manual_seed(1)
    m = torch.randn(horizon, horizon, generator=g, dtype=torch.float64) * 0.3 + torch.eye(
        horizon, dtype=torch.float64
    )
    x_vp = torch.randn(1, horizon, 2, generator=g, dtype=torch.float64)
    leftover = torch.randn(1, horizon, 2, generator=g, dtype=torch.float64)
    weights = (
        RTCProcessor(rtc_cfg(RTCAttentionSchedule.EXP)).get_prefix_weights(4, 10, horizon).to(torch.float64)
    )

    def x0_of(x):  # nonlinear denoiser in VP variables
        return torch.tanh(torch.einsum("st,bta->bsa", m, x))

    time, s = _vp_to_lerobot_flow(alpha_bar)
    tau = 1 - time  # paper convention
    a_tau = s * x_vp

    def velocity(a):  # A1_hat = A_tau + (1 - tau) v
        return (x0_of(a / s) - a) / (1 - tau)

    with torch.enable_grad():
        a = a_tau.clone().requires_grad_(True)
        a1 = a + (1 - tau) * velocity(a)
        err = (leftover - a1.detach()) * weights[None, :, None]
        (vjp,) = torch.autograd.grad(a1, a, grad_outputs=err)
    r2 = (1 - tau) ** 2 / (tau**2 + (1 - tau) ** 2)
    w_paper = min(beta, (1 - tau) / (tau * r2))
    delta_paper = (1 - tau) * w_paper * vjp  # change of A1_hat

    torch.testing.assert_close(
        _diffusion_x0_delta(x0_of, x_vp, leftover, weights, alpha_bar, beta, "flow"),
        delta_paper,
        rtol=1e-9,
        atol=1e-10,
    )
    if all(_clip_regime(alpha_bar, beta)):
        torch.testing.assert_close(
            _diffusion_x0_delta(x0_of, x_vp, leftover, weights, alpha_bar, beta, "pigdm"),
            delta_paper,
            rtol=1e-9,
            atol=1e-10,
        )


@pytest.mark.parametrize("alpha_bar", [0.0195, 0.478, 0.967, 0.9994])
@pytest.mark.parametrize("beta", [1.5, 10.0, 1e6])
@pytest.mark.parametrize("schedule", [RTCAttentionSchedule.EXP, RTCAttentionSchedule.LINEAR])
def test_identity_jacobian_reproduces_lerobot_rtc_processor(alpha_bar, beta, schedule):
    """`rtc_guidance_jacobian="identity"` + flow clip == LeRobot's RTCProcessor, for a nonlinear model."""
    horizon, delay, eh = 16, 3, 12
    processor = RTCProcessor(rtc_cfg(schedule, beta=beta, execution_horizon=eh))
    g = torch.Generator().manual_seed(2)
    m = torch.randn(horizon, horizon, generator=g, dtype=torch.float64) * 0.5
    x_vp = torch.randn(1, horizon, 2, generator=g, dtype=torch.float64)
    leftover = torch.randn(1, horizon, 2, generator=g, dtype=torch.float64)
    weights = processor.get_prefix_weights(delay, eh, horizon).to(torch.float64)

    def x0_of(x):
        return torch.tanh(torch.einsum("st,bta->bsa", m, x)) + 0.1 * x

    x0 = x0_of(x_vp)
    err = (leftover - x0) * weights[None, :, None]
    w = guidance_weight(alpha_bar, beta, "flow")
    delta = apply_guidance_correction(x0, flow_scale(alpha_bar) * err, alpha_bar, w, "sample") - x0

    time, s = _vp_to_lerobot_flow(alpha_bar)
    assert s == pytest.approx(flow_scale(alpha_bar))

    def velocity(xf):
        return (xf - x0_of(xf / s)) / time

    x_flow = s * x_vp
    v_guided = processor.denoise_step(x_flow, leftover, delay, time, velocity, execution_horizon=eh)
    delta_flow = -time * (v_guided - velocity(x_flow))
    torch.testing.assert_close(delta, delta_flow, rtol=2e-5, atol=1e-6)


# ------------------------------------------------------------------------------------------------
# Observation queue semantics, schedulers, warnings, debug
# ------------------------------------------------------------------------------------------------


def test_observe_and_snapshot_semantics():
    """Each control step pushes one observation; inference uses the snapshot at its start."""
    policy = make_policy(rtc_cfg())
    obs_seq = [make_obs(1, seed=s) for s in range(4)]
    noise = make_noise(policy, 1)

    policy.observe(obs_seq[0])
    policy.observe(obs_seq[1])
    a = policy.predict_action_chunk(obs_seq[2], noise=noise.clone())  # pushes obs 2 -> history (1, 2)
    assert torch.equal(policy._queues[OBS_STATE][0], obs_seq[1][OBS_STATE])
    assert torch.equal(policy._queues[OBS_STATE][1], obs_seq[2][OBS_STATE])

    ref = make_policy(rtc_cfg())
    ref.observe(obs_seq[1])  # first push fills the queue with copies, then obs 2 pushed below
    ref.observe(obs_seq[2])
    b = ref.predict_action_chunk(obs_seq[2], noise=noise.clone(), push_observation=False)
    assert torch.equal(a, b)

    with pytest.raises(RuntimeError, match="empty"):
        make_policy(rtc_cfg()).predict_action_chunk(obs_seq[0], push_observation=False)


def test_select_action_raises_with_rtc():
    policy = make_policy(rtc_cfg())
    with pytest.raises(RuntimeError, match="select_action"):
        policy.select_action(make_obs(1))


def test_ddpm_checkpoint_sampled_with_ddim_by_config_change():
    ddpm = make_policy(None, noise_scheduler_type="DDPM", num_inference_steps=None)
    ddim = make_policy(None, noise_scheduler_type="DDIM", num_inference_steps=10)
    ddim.load_state_dict(ddpm.state_dict())  # same weights, only the sampler differs
    set_inference_noise_scheduler(ddpm, "DDIM", 10)
    obs, noise = make_obs(2), make_noise(ddpm, 2)
    for p in (ddpm, ddim):
        p._queues = populate_queues(p._queues, dict(obs))
    assert torch.equal(
        ddpm.predict_action_chunk(dict(obs), noise=noise.clone()),
        ddim.predict_action_chunk(dict(obs), noise=noise.clone()),
    )
    assert ddpm.diffusion.num_inference_steps == 10


def test_many_steps_guidance_warns(caplog):
    with caplog.at_level(logging.WARNING):
        make_policy(rtc_cfg(), noise_scheduler_type="DDPM", num_inference_steps=None)
    assert any("VJP" in r.message for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        make_policy(rtc_cfg(), noise_scheduler_type="DDIM", num_inference_steps=10)
        make_policy(
            rtc_cfg(), noise_scheduler_type="DDPM", num_inference_steps=None, rtc_guidance_jacobian="identity"
        )
    assert not any("VJP" in r.message for r in caplog.records)


def test_init_rtc_processor_after_construction():
    """lerobot-rollout sets `config.rtc_config` on a loaded policy, then calls `init_rtc_processor`."""
    policy = make_policy(None)
    assert policy.rtc_processor is None
    policy.config.rtc_config = rtc_cfg(RTCAttentionSchedule.EXP)
    policy.init_rtc_processor()
    assert policy.rtc_processor is not None
    actions = policy.predict_action_chunk(
        make_obs(1), inference_delay=2, prev_chunk_left_over=torch.rand(10, 2)
    )
    assert actions.shape == (1, 15, ACTION_DIM)


def test_debug_tracker_records_each_step():
    policy = make_policy(rtc_cfg(RTCAttentionSchedule.EXP, debug=True, debug_maxlen=1000))
    policy.predict_action_chunk(make_obs(1), inference_delay=3, prev_chunk_left_over=torch.rand(10, 2))
    steps = policy.rtc_processor.get_all_debug_steps()
    assert len(steps) == policy.config.num_inference_steps
    for step in steps:
        assert step.correction is not None and step.err is not None and step.x1_t is not None
        assert step.guidance_weight is not None and step.inference_delay == 3 and step.execution_horizon == 10
        for key in ("alpha_bar", "err_norm", "grad_norm", "delta_out_norm", "delta_x0_norm", "timestep"):
            assert key in step.metadata
    assert len(policy.last_rtc_info["steps"]) == policy.config.num_inference_steps


# ------------------------------------------------------------------------------------------------
# r2 = Gaussian posterior variance (rtc_guidance_r2="posterior", experimental)
# ------------------------------------------------------------------------------------------------

DIAG_VAR = 0.25


def diagonal_gaussian_denoiser(policy: DiffusionPolicy, prediction_type: str) -> GaussianPriorDenoiser:
    """Exact denoiser for x0 ~ N(0, DIAG_VAR I) (RBF lengthscale -> 0: independent dims)."""
    den = GaussianPriorDenoiser(
        policy.diffusion.noise_scheduler,
        policy.config.horizon,
        prediction_type,
        var=DIAG_VAR,
        lengthscale=1e-3,
    )
    return den.double()


@pytest.mark.parametrize("timestep", [90, 50, 10, 0])
def test_posterior_r2_makes_jacobian_ratio_exact_for_diagonal_gaussian_prior(timestep):
    """x0_d ~ N(0, s2): J_dd / r2_d == sqrt(a) / (1 - a), while J_dd / (1 - a) (unit r2) is off unless
    s2 >> (1 - a) / a (at the last DDIM step J ~ sqrt(a) for any s2, so both agree there)."""
    policy = make_policy(None)
    den = diagonal_gaussian_denoiser(policy, "sample")
    a = float(policy.diffusion.noise_scheduler.alphas_cumprod[timestep])
    x = torch.randn(2, policy.config.horizon, ACTION_DIM, dtype=torch.float64, requires_grad=True)
    x0 = den.x0(x, torch.full((2,), timestep))
    (j_diag,) = torch.autograd.grad(x0, x, grad_outputs=torch.ones_like(x0))  # J is diagonal
    s2 = torch.full_like(x0, DIAG_VAR + 1e-6)  # + the jitter added to the kernel
    expected = torch.full_like(x0, a**0.5 / (1 - a))
    torch.testing.assert_close(j_diag / posterior_r2(a, s2), expected, rtol=1e-6, atol=0)
    if timestep >= 10:
        assert not torch.allclose(j_diag / (1 - a), expected, rtol=1e-2)


@pytest.mark.parametrize("prediction_type", ["epsilon", "sample"])
@pytest.mark.parametrize("r2_clip", ["relative", "literal"])
@pytest.mark.parametrize("clip", ["pigdm", "flow"])
def test_posterior_r2_unclipped_moves_x0_hat_onto_target(prediction_type, r2_clip, clip):
    """Diagonal Gaussian prior, exact s2, no clipping: every guided step sets x0_hat = Y on W == 1 slots."""
    policy = make_policy(None, prediction_type=prediction_type)
    model = policy.diffusion
    model.unet = diagonal_gaussian_denoiser(policy, prediction_type)
    model.noise_scheduler.set_timesteps(10)
    g = torch.Generator().manual_seed(0)
    target = 0.3 * torch.randn(2, 16, ACTION_DIM, generator=g, dtype=torch.float64)
    weights = torch.zeros(16, dtype=torch.float64)
    weights[1:6] = 1.0
    guide = DiffusionRTCGuide(
        mode="guidance",
        target=target,
        weights=weights,
        max_guidance_weight=1e12,
        prediction_type=prediction_type,
        processor=RTCProcessor(rtc_cfg(debug=True, debug_maxlen=100)),
        weight_clip=clip,
        prior_var=torch.full_like(target, DIAG_VAR + 1e-6),
        r2_clip=r2_clip,
    )
    sample = torch.randn(2, 16, ACTION_DIM, generator=g, dtype=torch.float64)
    for t in model.noise_scheduler.timesteps:
        guide._guided_step(model, sample, None, t, None)
        stats = guide.step_stats[-1]
        assert stats["corr_frac"] == pytest.approx(1.0, rel=1e-6)
        assert stats["delta_x0_norm"] == pytest.approx(stats["err_norm"], rel=1e-6)


def test_unit_r2_misses_target_for_diagonal_gaussian_prior():
    """Same setting with the paper's r2 = 1 - a: the correction is J / (1 - a) e, i.e. too weak by ~s2."""
    policy = make_policy(None)
    model = policy.diffusion
    model.unet = diagonal_gaussian_denoiser(policy, "epsilon")
    model.noise_scheduler.set_timesteps(10)
    target = torch.full((1, 16, ACTION_DIM), 0.3, dtype=torch.float64)
    weights = torch.ones(16, dtype=torch.float64)
    guide = DiffusionRTCGuide(
        mode="guidance",
        target=target,
        weights=weights,
        max_guidance_weight=1e12,
        prediction_type="epsilon",
        processor=RTCProcessor(rtc_cfg(debug=True, debug_maxlen=100)),
    )
    sample = torch.randn(1, 16, ACTION_DIM, dtype=torch.float64)
    for t in model.noise_scheduler.timesteps:
        guide._guided_step(model, sample, None, t, None)
        a = float(model.noise_scheduler.alphas_cumprod[int(t)])
        s2 = DIAG_VAR + 1e-6
        expected = s2 / (a * s2 + 1 - a)  # J / (1 - a) * (1 - a) / sqrt(a) = s2 / (a s2 + 1 - a)
        assert guide.step_stats[-1]["corr_frac"] == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("clip", ["pigdm", "flow"])
@pytest.mark.parametrize("alpha_bar", [0.02, 0.5, 0.999])
def test_posterior_guidance_weight_formulas(clip, alpha_bar):
    s2 = torch.tensor([1e-4, 1e-2, 0.25, 1.0])
    r2 = posterior_r2(alpha_bar, s2)
    torch.testing.assert_close(r2, s2 * (1 - alpha_bar) / (alpha_bar * s2 + 1 - alpha_bar))
    torch.testing.assert_close(posterior_r2(alpha_bar, torch.ones(1)), torch.tensor([1 - alpha_bar]))
    for beta in (1.0, 10.0, 1e12):
        rel = posterior_guidance_weights(alpha_bar, s2, beta, clip, "relative")
        torch.testing.assert_close(rel, guidance_weight(alpha_bar, beta, clip) * (1 - alpha_bar) / r2)
        lit = posterior_guidance_weights(alpha_bar, s2, beta, clip, "literal")
        if clip == "pigdm":
            expected = torch.clamp(1 / r2, max=beta)
        else:
            c = ((1 - alpha_bar) / alpha_bar) ** 0.5
            expected = (alpha_bar / (1 - alpha_bar)) ** 0.5 * torch.clamp(c / r2, max=beta)
        torch.testing.assert_close(lit, expected)
    unclipped = 1 / r2
    for r2_clip in ("relative", "literal"):
        torch.testing.assert_close(posterior_guidance_weights(alpha_bar, s2, 1e12, clip, r2_clip), unclipped)


@pytest.mark.parametrize("clip", ["pigdm", "flow"])
@pytest.mark.parametrize("jacobian", ["vjp", "identity"])
def test_posterior_r2_with_unit_variance_equals_unit_r2(clip, jacobian):
    """s2 = 1 and the "relative" clip reproduce the paper's r2 = 1 - a path."""
    policy = make_policy(
        rtc_cfg(RTCAttentionSchedule.EXP, beta=5.0),
        rtc_guidance_weight_clip=clip,
        rtc_guidance_jacobian=jacobian,
    )
    model = policy.diffusion
    obs, noise = make_obs(2), make_noise(policy, 2)
    leftover = 2 * torch.rand(2, 10, ACTION_DIM, generator=torch.Generator().manual_seed(3)) - 1
    policy.observe(obs)
    global_cond = model._prepare_global_conditioning(snapshot_observations(policy))
    kwargs = {"batch_size": 2, "inference_delay": 3, "prev_chunk_left_over": leftover}
    unit = build_guide(policy, **kwargs)
    post = build_guide(policy, **kwargs)
    post.prior_var = torch.ones_like(post.target)
    expected = model.conditional_sample(2, global_cond=global_cond, noise=noise.clone(), rtc_guide=unit)
    actual = model.conditional_sample(2, global_cond=global_cond, noise=noise.clone(), rtc_guide=post)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_posterior_r2_end_to_end():
    # "literal" keeps 1 / r2 <= beta. With "relative" a randomly initialised U-Net diverges: its Jacobian does
    # not scale like s2, so the 1 / r2 ~ 1 / s2 amplification is not compensated (see the config docstring).
    policy = make_policy(
        rtc_cfg(RTCAttentionSchedule.EXP),
        rtc_guidance_r2="posterior",
        rtc_r2_num_samples=4,
        rtc_r2_clip="literal",
    )
    obs, noise = make_obs(3), make_noise(policy, 3)
    leftover = 2 * torch.rand(3, 10, ACTION_DIM, generator=torch.Generator().manual_seed(3)) - 1
    kwargs = {"inference_delay": 2, "prev_chunk_left_over": leftover}

    rng_state = torch.get_rng_state()
    first = policy.predict_action_chunk(dict(obs), noise=noise.clone(), **kwargs)
    assert torch.equal(torch.get_rng_state(), rng_state)  # s2 sampling uses its own generator
    prior_var = policy.last_rtc_info["prior_var"]
    assert prior_var.shape == (3, policy.config.horizon, ACTION_DIM) and (prior_var > 0).all()
    assert first.shape == (3, 15, ACTION_DIM) and torch.isfinite(first).all()

    # Same s2 noise stream after init_rtc_processor -> identical output; also under inference_mode.
    policy.reset()
    policy.init_rtc_processor()
    with torch.inference_mode():
        again = policy.predict_action_chunk(dict(obs), noise=noise.clone(), **kwargs)
    torch.testing.assert_close(again, first)

    # Inactive guide (no leftover): no s2 estimation, original sampler.
    policy.reset()
    policy.predict_action_chunk(dict(obs), noise=noise.clone())
    assert policy.last_rtc_info["prior_var"] is None


# ------------------------------------------------------------------------------------------------
# lerobot-rollout RTC engine integration
# ------------------------------------------------------------------------------------------------


class _IdentityPipeline:
    steps: list = []

    def __call__(self, x):
        return x

    def reset(self):
        pass


def test_rollout_rtc_engine_feeds_per_tick_observation_history():
    import time
    from unittest.mock import MagicMock

    from lerobot.rollout.inference.rtc import RTCInferenceEngine

    policy = make_policy(None)
    policy.config.rtc_config = rtc_cfg(RTCAttentionSchedule.EXP)  # what lerobot-rollout does
    policy.init_rtc_processor()
    names_state, names_env = [f"s{i}" for i in range(STATE_DIM)], [f"e{i}" for i in range(ENV_DIM)]
    hw_features = {
        OBS_STATE: {"dtype": "float32", "shape": (STATE_DIM,), "names": names_state},
        "observation.environment_state": {"dtype": "float32", "shape": (ENV_DIM,), "names": names_env},
    }
    engine = RTCInferenceEngine(
        policy=policy,
        preprocessor=_IdentityPipeline(),
        postprocessor=_IdentityPipeline(),
        robot_wrapper=MagicMock(robot_type="mock"),
        rtc_config=policy.config.rtc_config,
        hw_features=hw_features,
        task="",
        fps=30.0,
        device="cpu",
    )
    ticks = [
        {**{n: float(10 * k + i) for i, n in enumerate(names_state)}, **dict.fromkeys(names_env, 0.0)}
        for k in (1, 2)
    ]
    engine.start()
    try:
        for obs in ticks:
            engine.notify_observation(obs)
        engine.resume()
        deadline = time.time() + 60
        while engine.action_queue.qsize() == 0 and time.time() < deadline and not engine.failed:
            time.sleep(0.01)
        engine.pause()
    finally:
        engine.stop()
    assert not engine.failed
    assert engine.action_queue.qsize() > 0
    history = [q[0].tolist() for q in policy._queues[OBS_STATE]]
    assert history == [[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]]  # two consecutive ticks, oldest first
