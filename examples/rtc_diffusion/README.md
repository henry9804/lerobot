# Real-Time Chunking for DiffusionPolicy

Inference-time RTC ([Black et al., 2025](https://arxiv.org/abs/2506.07339)) for `DiffusionPolicy`
(DDPM/DDIM, `epsilon` or `sample` prediction). Training is unchanged. Implementation:
`src/lerobot/policies/diffusion/rtc_diffusion.py`; tests: `tests/policies/diffusion/test_diffusion_rtc.py`.

## Usage

```python
from lerobot.configs import RTCAttentionSchedule
from lerobot.policies.rtc import ActionQueue, RTCConfig
from lerobot.policies.diffusion.rtc_diffusion import set_inference_noise_scheduler

rtc = RTCConfig(enabled=True, prefix_attention_schedule=RTCAttentionSchedule.EXP,
                execution_horizon=15, max_guidance_weight=10.0)
policy.config.rtc_config = rtc
policy.config.rtc_guidance_mode = "guidance"      # "none" | "guidance" | "replacement"
policy.init_rtc_processor()
set_inference_noise_scheduler(policy, "DDIM", 10)   # DDPM-trained checkpoints work as-is

queue = ActionQueue(rtc)
# every control step: push exactly one observation
#   - at steps where inference starts:  chunk = policy.predict_action_chunk(obs, inference_delay=d,
#                                              prev_chunk_left_over=queue.get_left_over())
#   - at all other steps:               policy.observe(obs)
original = chunk.squeeze(0)                       # normalized, (H_eff, A), H_eff = horizon - n_obs_steps + 1
processed = postprocessor(chunk).squeeze(0)       # env / robot units
queue.merge(original, processed, d)               # `original` MUST be the normalized chunk
```

`lerobot-rollout --inference.type=rtc --policy.path=<diffusion ckpt> --policy.noise_scheduler_type=DDIM
--policy.num_inference_steps=10 ...` also works (the engine now forwards per-tick observation history).
Set `--inference.queue_threshold` below `H_eff`, otherwise the engine re-plans continuously.

## Config fields (`DiffusionConfig`)

| field | default | meaning |
|---|---|---|
| `rtc_config` | `None` | `None` / `enabled=False` → original behaviour, bit-identical |
| `rtc_guidance_mode` | `"guidance"` | `none` (naive async), `guidance` (PiGDM), `replacement` (RePaint-style, W==1 slots) |
| `rtc_guidance_jacobian` | `"vjp"` | `vjp`: exact VJP through the U-Net (paper). `identity`: no backward; what LeRobot's flow `RTCProcessor` actually computes |
| `rtc_guidance_weight_clip` | `"pigdm"` | `pigdm`: `w=min(β, 1/(1-ᾱ))`. `flow`: LeRobot/paper flow clipping mapped to VP (same β meaning as pi0) |
| `rtc_constrain_past_actions` | `False` | also pin raw slots `[0, n_obs_steps-1)` to `prev_executed_actions` |
| `rtc_guidance_r2` | `"unit"` | PiGDM `r²`. `unit`: `1-ᾱ` (paper). `posterior` (experimental): per-element `s²(1-ᾱ)/(ᾱs²+1-ᾱ)`, `s²` from `rtc_r2_num_samples` unguided chunks of the same observation |
| `rtc_r2_num_samples` | `16` | chunks for the `s²` estimate (`posterior` only) |
| `rtc_r2_clip` | `"relative"` | `posterior` only. `relative`: β clips the unit-prior weight, `(1-ᾱ)/r²_d` applied on top. `literal`: β clips `1/r²_d` |

## Coordinates and naming

* `prev_chunk_left_over[i]` ↔ executable index `i` of the new chunk ↔ raw horizon index `n_obs_steps-1+i`
  ↔ time `t+i` (`t` = time of the latest observation). See `exec_to_raw_index`.
* `RTCConfig.execution_horizon` = overlap length (paper `H−s`), called `overlap_len` in code. The paper's
  `s` is called `replan_every` in the scripts. `overlap_len = min(execution_horizon, len(leftover), H_eff)`,
  `inference_delay ≤ overlap_len`.

## Scripts

```bash
# offline: chunk A at frame k, chunk B at k+s with leftover A[s:], per variant; plots + CSV
python examples/rtc_diffusion/eval_dataset.py --policy.path=<ckpt> \
  --policy.noise_scheduler_type=DDIM --policy.num_inference_steps=10 \
  --dataset.repo_id=lerobot/pusht --inference_delay=3 --replan_every=8 --num_samples=16

# deterministic simulated latency (no threads): sync / naive / guidance / replacement, sweep d, s, beta
python examples/rtc_diffusion/eval_sim_delay.py --policy.path=<ckpt> \
  --policy.noise_scheduler_type=DDIM --policy.num_inference_steps=10 --env.type=pusht \
  --n_episodes=50 --batch_size=10 --delays='[0,1,2,3,4,6]' --replan_every='[8]' --betas='[10.0]'

# is J / r2 = sqrt(a)/(1-a)? learned Jacobian vs unit / Gaussian-posterior r2 vs the model's own
# ancestral posterior covariance, per denoising step
python examples/rtc_diffusion/diag_jacobian_r2.py --policy.path=<ckpt> \
  --policy.noise_scheduler_type=DDIM --policy.num_inference_steps=10 \
  --dataset.repo_id=lerobot/pusht --num_frames=24 --num_var_samples=64 --num_posterior_samples=64
```

`eval_dataset.py` / `eval_sim_delay.py` also accept the `guidance_exp_post` (posterior r², relative clip)
and `guidance_exp_post_lit` (literal clip) variants; they are not in the default lists.

## Caveats (measured on `lerobot/diffusion_pusht`, DDIM-10)

* With the paper-faithful VJP, guidance is weak for policies with a narrow conditional action
  distribution: the Tweedie Jacobian `∂x̂0/∂x_t ≈ √ᾱ·Cov[x0|x_t]/(1−ᾱ)` is ~0, while PiGDM assumes a
  unit-variance prior (`r² = 1−ᾱ`). `identity` + `flow` clip (LeRobot-equivalent) is much stronger and has
  no extra cost; VJP guidance costs ~1.6–1.8× the unguided latency.
* β is not comparable across clip modes: at `t=0` the `flow` clip gives an effective weight ≈ √(ᾱ/(1−ᾱ))·β
  (≈ 40β), the `pigdm` clip gives β. Sweep β per clip mode.
* Correcting `r²` does not fix VJP guidance (`diag_jacobian_r2.py`, 24 frames). The conditional action variance is
  tiny (median `s² ≈ 3e-5`), and the Gaussian-posterior `r²` matches the model's own ancestral posterior variance
  (ratio 0.9–1.05 at every step). The learned Jacobian does not match it: `J/r²_post` divided by `√ᾱ/(1-ᾱ)` has a
  random sign and is 2–100× off at steps 0–7. `J` is also strongly asymmetric (0.5–0.66; the exact Tweedie
  Jacobian is symmetric), with the same values in fp64. With epsilon prediction, `J` is the residual of a
  near-identity map, `√(1-ᾱ)∂ε/∂x = I − √ᾱJ`. The network's residual is about 1% of the identity, while the
  true residual is about 1e-6 to 3e-3, so `J` is mostly approximation error. With `1/r²_post ~ 1/s²`, the
  posterior variant amplifies that error and diverges (`relative`, or `literal` with a large β). With a small β
  it reduces to the unit-r² behaviour.
* lerobot-rollout pads/truncates the leftover to `execution_horizon` with zeros; when the real leftover is
  shorter, the zero rows are treated as targets (pre-existing behaviour, also for pi0/smolvla).
