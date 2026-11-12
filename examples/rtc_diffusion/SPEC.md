# DiffusionPolicy용 Real-Time Chunking: 구현 스펙

- 기준 코드: `main` @ `da92db8f`와 커밋하지 않은 작업 트리 (2026-11-09)
- 논문: Black et al., *Real-Time Execution of Action Chunking Flow Policies*, arXiv:2506.07339 (2025)
- 사용법만 필요하면 [README.md](README.md)를 보세요. 이 문서는 무엇을 어떻게 구현했는지, 어떤 가정을 두었는지, 무엇으로 검증했는지를 정리합니다.

## 0. 요약

| 항목 | 내용 |
|---|---|
| 목적 | DiffusionPolicy(DDPM/DDIM, `epsilon`/`sample` prediction)의 비동기 추론에 RTC를 연구용 baseline으로 추가 |
| 핵심 동작 | 새 chunk를 생성하는 동안, 이전 chunk 중 어차피 실행될 부분(prefix)을 inpainting으로 강제 |
| 모드 | `guidance`(ΠGDM, 논문 방식) · `replacement`(RePaint식 hard inpainting) · `none`(naive async) |
| guidance 변형 축 | Jacobian(`vjp` / `identity`) × weight clip(`pigdm` / `flow`) × r²(`unit` / `posterior`) |
| 학습 | 변경 없음. 기존 체크포인트를 그대로 사용하고, DDPM으로 학습한 모델을 DDIM으로 추론해도 됨 |
| 호환성 | `rtc_config=None` 또는 `enabled=False`이면 기존 출력과 bit 단위로 동일(테스트로 확인) |
| PushT 결론 | 논문식 VJP guidance는 효과가 거의 없음. `identity` + `flow`(LeRobot pi0 RTC와 같은 계산)가 비용 증가 없이 효과적. r² 보정 가설은 기각(§11.4) |

## 1. 설계 원칙과 범위

1. **학습 코드와 loss는 건드리지 않습니다.** RTC는 추론 시점의 reverse process만 바꿉니다.
2. **기존 diffusion 파일의 diff를 최소화합니다.** 로직은 전부 `policies/diffusion/rtc_diffusion.py`에 두고, `modeling_diffusion.py`에는 hook만 추가했습니다.
3. **LeRobot API를 추측해서 쓰지 않습니다.** prefix 가중치(`RTCProcessor.get_prefix_weights`), debug tracker, `ActionQueue`는 LeRobot 구현을 그대로 재사용하고, 동작이 같은지는 테스트로 대조했습니다(§9).
4. **확인하지 못한 가정은 코드 주석과 이 문서에 명시합니다(§8).**

지원 범위:

| 항목 | 지원 여부 |
|---|---|
| scheduler | DDPM, DDIM (`replacement` 모드에서 그 외 scheduler는 `TypeError`) |
| prediction type | `epsilon`, `sample`. `v_prediction`은 `ValueError` |
| 진입점 | `predict_action_chunk` + `observe`. RTC가 켜져 있으면 `select_action`은 `RuntimeError` |
| 실행기 | 직접 호출, `lerobot-rollout --inference.type=rtc`, 예제 평가 스크립트 |
| 검증 안 함 | `compile_model=True` + VJP, 실제 로봇 rollout |

## 2. 변경 및 추가 파일

| 파일 | 종류 | 내용 |
|---|---|---|
| `src/lerobot/policies/diffusion/rtc_diffusion.py` | 신규 | index mapping, guidance 수식, guided/replacement 루프, σ² 추정, policy-level helper |
| `src/lerobot/policies/diffusion/configuration_diffusion.py` | 수정 | RTC 설정 필드 8개와 검증 |
| `src/lerobot/policies/diffusion/modeling_diffusion.py` | 수정 | `init_rtc_processor`, `observe`, `predict_action_chunk` kwargs, `select_action` 가드, `conditional_sample(rtc_guide=...)` hook |
| `src/lerobot/rollout/inference/rtc.py` | 수정 | `n_obs_steps > 1` 정책을 위해 control tick별 관측 history를 정책에 전달 |
| `tests/policies/diffusion/test_diffusion_rtc.py` | 신규 | 252개 케이스 |
| `examples/rtc_diffusion/` | 신규 | `README.md`, `SPEC.md`(이 문서), `eval_dataset.py`, `eval_sim_delay.py`, `diag_jacobian_r2.py` |

작업 트리의 다른 수정 파일(`pyproject.toml`, pi0/pi05/smolvla 등)은 이 작업 이전부터 있던 변경이며 RTC와 관계없습니다.

## 3. 용어와 좌표계

좌표계는 가장 실수하기 쉬운 부분이라, 모든 변환을 `exec_to_raw_index` 한 곳에서 합니다.

| 기호 | 정의 |
|---|---|
| `H` | `config.horizon` (모델이 예측하는 action 수, raw horizon) |
| `n_obs` | `config.n_obs_steps` |
| `H_eff` | `H − n_obs + 1`. 실행 가능한 action 수. PushT는 16 − 2 + 1 = 15 |
| `A` | action 차원 |
| `t` | snapshot에서 가장 최근 관측의 시각 |

- **raw index** `j`는 시각 `t − (n_obs − 1) + j`의 action입니다.
- **executable index** `i`는 raw index `n_obs − 1 + i`, 즉 시각 `t + i`입니다. RTC 모드의 `predict_action_chunk`는 executable 범위 전체 `(B, H_eff, A)`를 반환합니다.
- `prev_chunk_left_over[i]`는 이전 chunk가 시각 `t + i`에 내놓은 action입니다. 새 chunk의 executable index `i`와 정렬되며, `ActionQueue.get_left_over()`와 같은 규약입니다.
- leftover, guidance target, 반환값은 모두 **정규화된 action 공간**(postprocessor 이전)입니다. `ActionQueue.merge(original, processed, d)`의 `original`에는 반드시 정규화된 chunk를 넣어야 합니다.

이름 대응:

| 논문 | LeRobot | 이 구현 |
|---|---|---|
| `H − s` (이전 chunk와 일관성을 유지하는 구간) | `RTCConfig.execution_horizon` | `overlap_len` |
| `s` (inference 시작 간격) | 없음 | `replan_every` (평가 스크립트) |
| `d` (추론 지연) | `inference_delay` | `inference_delay` |
| `β` | `RTCConfig.max_guidance_weight` | 같음 |

`resolve_overlap`이 다음과 같이 clamp합니다.

```
overlap_len     = min(execution_horizon, len(leftover), H_eff)
inference_delay = clamp(d, 0, overlap_len)        # None → 0
```

## 4. 호출 프로토콜과 API

### 4.1 관측 push 규칙

**매 control step마다 관측을 정확히 하나씩 push해야 합니다.** inference를 시작하는 step에서는 `predict_action_chunk(obs, ...)`가 push하고, 나머지 step에서는 `policy.observe(obs)`를 호출합니다. 그래야 `n_obs_steps` history가 control step 단위로 연속됩니다.

```python
from lerobot.configs import RTCAttentionSchedule
from lerobot.policies.rtc import ActionQueue, RTCConfig
from lerobot.policies.diffusion.rtc_diffusion import set_inference_noise_scheduler

policy.config.rtc_config = RTCConfig(enabled=True, prefix_attention_schedule=RTCAttentionSchedule.EXP,
                                     execution_horizon=15, max_guidance_weight=10.0)
policy.config.rtc_guidance_mode = "guidance"
policy.init_rtc_processor()
set_inference_noise_scheduler(policy, "DDIM", 10)

queue = ActionQueue(policy.config.rtc_config)
# inference 시작 step:
chunk = policy.predict_action_chunk(obs, inference_delay=d, prev_chunk_left_over=queue.get_left_over())
queue.merge(chunk.squeeze(0), postprocessor(chunk).squeeze(0), d)
# 그 외 step:
policy.observe(obs)
```

### 4.2 `DiffusionPolicy` 공개 API

| API | 동작 |
|---|---|
| `predict_action_chunk(batch, noise=None, inference_delay=None, prev_chunk_left_over=None, *, execution_horizon=None, prev_executed_actions=None, push_observation=True, generator=None)` | RTC가 켜져 있으면 `rtc_predict_action_chunk`로 위임하고 `(B, H_eff, A)`를 반환. 그 외 kwargs는 무시(debug log만 남김). 꺼져 있으면 기존 경로 |
| `observe(batch)` | inference 없이 관측만 queue에 push. `select_action`과 같은 전처리(`ACTION` 제거, 카메라 stack, 첫 호출 때 queue 채우기)를 하고, 호출자의 dict는 수정하지 않음 |
| `init_rtc_processor()` | `config.rtc_config`로 `RTCProcessor`를 다시 만들고 σ² 샘플 카운터를 0으로 리셋. `__init__`에서 한 번 호출되며, 생성 후 `rtc_config`를 바꿨다면 다시 호출해야 함(`lerobot-rollout`이 이렇게 함) |
| `select_action` | RTC가 켜져 있으면 `RuntimeError` |
| `last_rtc_info` | 마지막 호출 정보: `mode`, `active`, `inference_delay`, `overlap_len`, `leftover_len`, `weights`(H,), `steps`(step별 통계, §7), `prior_var` |

인자 형식:

- `prev_chunk_left_over`와 `prev_executed_actions`는 `(T, A)` 또는 `(B, T, A)`입니다. batch 1은 B로 broadcast되고, action 차원이나 batch가 맞지 않으면 `ValueError`가 납니다.
- `prev_executed_actions`는 시각 `t − 1`까지 실제로 실행된 정규화 action이며, 마지막 행이 `t − 1`입니다. `rtc_constrain_past_actions=True`일 때만 쓰입니다.
- `push_observation=False`는 현재 관측을 이미 `observe`로 넣은 경우에 씁니다.

### 4.3 helper (`rtc_diffusion.py`)

- `set_inference_noise_scheduler(policy, "DDIM", 10)`: 학습 없이 sampler만 교체합니다. CLI의 `--policy.noise_scheduler_type=DDIM --policy.num_inference_steps=10`과 같습니다.
- `snapshot_observations(policy)`: queue를 `(B, n_obs, ...)`로 stack합니다. queue가 비어 있으면 `RuntimeError`가 납니다.
- 관측 queue는 process 전역 `RLock` 하나로 보호합니다. lock을 policy에 두면 `copy.deepcopy(policy)`가 깨지기 때문입니다.

### 4.4 `lerobot-rollout` 연동 (`rollout/inference/rtc.py`)

- 기존 엔진은 inference를 시작할 때의 관측 하나만 정책에 넘겼습니다. 그래서 `n_obs_steps > 1`이면 history가 control tick 단위로 이어지지 않았습니다.
- 변경 후에는 `n_obs_steps > 1`이고 정책에 `observe`가 있을 때, 엔진이 `deque(maxlen=n_obs_steps)`에 tick별 관측을 쌓습니다.
- inference 직전에 과거 tick(`history[:-1]`)을 오래된 순서로 `observe`에 다시 넣고, 현재 관측은 `predict_action_chunk`가 push합니다. reset 때 history도 비웁니다.
- 전처리 코드는 `_preprocess_observation`으로 분리만 했고 동작은 그대로입니다. `n_obs_steps == 1`인 정책은 이전과 같습니다.

## 5. 설정 (`DiffusionConfig`)

| 필드 | 기본값 | 값 | 의미 |
|---|---|---|---|
| `rtc_config` | `None` | `RTCConfig` | `None` 또는 `enabled=False`이면 기존 동작(bit 동일) |
| `rtc_guidance_mode` | `"guidance"` | `none` / `guidance` / `replacement` | prefix 강제 방식 |
| `rtc_constrain_past_actions` | `False` | bool | raw slot `[0, n_obs−1)`도 `prev_executed_actions`로 고정(가중치 1) |
| `rtc_guidance_weight_clip` | `"pigdm"` | `pigdm` / `flow` | β로 가중치를 clip하는 방식(§6.2) |
| `rtc_guidance_jacobian` | `"vjp"` | `vjp` / `identity` | VJP 계산 방식(§6.2) |
| `rtc_guidance_r2` | `"unit"` | `unit` / `posterior` | ΠGDM의 r². `posterior`는 실험용(§6.3) |
| `rtc_r2_num_samples` | `16` | int ≥ 2 | σ² 추정에 쓰는 chunk 수 K |
| `rtc_r2_clip` | `"relative"` | `relative` / `literal` | `posterior`에서 β의 적용 방식 |

- 잘못된 값은 `__post_init__`에서 `ValueError`로 거부합니다.
- 이 필드가 없는 기존 `config.json`도 그대로 로드됩니다(테스트로 확인).
- `RTCConfig`에서 쓰는 필드는 `enabled`, `prefix_attention_schedule`, `execution_horizon`, `max_guidance_weight`, `debug`, `debug_maxlen`입니다.

## 6. 알고리즘

### 6.1 prefix 목표와 가중치

1. executable 좌표의 가중치는 `RTCProcessor.get_prefix_weights(d, overlap_len, H_eff)`로 만듭니다(LeRobot 구현 재사용).

   | schedule | `[0, d)` | `[d, overlap_len)` | `[overlap_len, H_eff)` |
   |---|---|---|---|
   | `ZEROS` | 1 | 0 | 0 |
   | `ONES` | 1 | 1 | 0 |
   | `LINEAR` | 1 | 1에서 0으로 선형 감소(양 끝 제외) | 0 |
   | `EXP` | 1 | `u · expm1(u) / (e − 1)`, `u`는 LINEAR 값 | 0 |

2. `build_raw_prefix`가 이를 raw horizon에 배치합니다. target `Y`는 `(B, H, A)`, 가중치 `W`는 `(H,)`이고, leftover는 raw `[n_obs−1, n_obs−1+n)`에 놓입니다. 나머지 위치는 `Y = 0`, `W = 0`입니다.
3. `rtc_constrain_past_actions=True`이면 raw `[0, n_obs−1)`에 `prev_executed_actions`를 오른쪽 정렬로 넣고 `W = 1`로 둡니다.

### 6.2 guidance 모드 (ΠGDM)

`DiffusionModel.conditional_sample`은 `scheduler.set_timesteps` 직후 `rtc_guide.denoise(...)`로 넘어갑니다. denoising step마다(`a = alphas_cumprod[t]`):

```
x0_hat = (x_t − √(1−a)·ε) / √a              (epsilon)   |   x0_hat = out   (sample)
e      = W ⊙ (Y − x0_hat)
g      = Jᵀ c,  J = ∂x0_hat/∂x_t,  c = e (unit r²)  또는  c = λ ⊙ e (posterior r², §6.3)
ε'     = ε − √(1−a) · w · g                 (epsilon)
x0'    = out + (1−a)/√a · w · g             (sample)
x_{t−1} = scheduler.step(out', t, x_t)      # clip_sample 등 scheduler 처리는 그대로
```

- 두 parameterization에서 x0_hat의 변화량은 `Δx0_hat = (1−a)/√a · w · g`로 같습니다(테스트로 확인).
- clip 없이 unit r²(`w = 1/(1−a)`)를 쓰면 `Δx0_hat = g/√a`입니다.

**가중치 `w` (unit r², `r² = 1 − a`):**

| clip | 식 | 특징 |
|---|---|---|
| `pigdm` | `w = min(β, 1/(1−a))` | 저잡음 끝만 clip |
| `flow` | `w = √(a/(1−a)) · min(β, 1/√(a(1−a)))` | LeRobot flow `RTCProcessor`의 `min(β, (1−τ)/(τ r_τ²))`를 VP 변수로 옮긴 식. 양 끝 모두 clip |

clip이 걸리지 않으면 두 방식 모두 `1/(1−a)`입니다. 하지만 같은 β라도 의미가 다릅니다. 저잡음 끝(a ≈ 0.9994)에서 `flow`의 실효 가중치는 약 `√(a/(1−a))·β ≈ 40β`입니다. 따라서 β는 clip 방식마다 따로 sweep해야 합니다.

**Jacobian:**

| 값 | 계산 | 비용 |
|---|---|---|
| `vjp` (기본, 논문) | `torch.autograd.grad(x0_hat, x_t, grad_outputs=c)`. U-Net을 통과하는 정확한 VJP | step마다 forward와 backward 1회 |
| `identity` | `g = s · c`, `s = 1/(√a + √(1−a))` (`flow_scale`) | forward만. unguided와 같은 비용 |

- `identity`는 LeRobot flow `RTCProcessor`가 실제로 계산하는 것과 같습니다. 그 구현은 `x_t.requires_grad_(True)` 전에 velocity를 계산하므로, VJP가 사실상 rectified-flow 좌표의 identity Jacobian으로 계산됩니다. 테스트에서 비선형 denoiser로 `RTCProcessor.denoise_step`과 수치 일치를 확인했습니다.
- `identity` + 무clip에서는 `Δx0_hat = (s/√a) · e`입니다. 고잡음 step에서는 과보정됩니다(a = 0.0195에서 6.3배).

**autograd 안전장치 (`vjp`):**

- 입력 중 `x_t`만 `requires_grad`이므로 모델 파라미터에는 `.grad`가 생기지 않습니다.
- 호출자가 `torch.inference_mode` 안에 있을 수 있어서(`lerobot-eval` 등) 루프를 `torch.inference_mode(False)`로 감싸고, `sample`과 `global_cond`를 clone합니다.
- VJP guidance에 inference step이 20을 넘으면 warning을 냅니다. DDPM-100은 첫 step의 ᾱ가 약 2.4e-7이어서 Jacobian이 `1/√a`로 증폭됩니다.
- `compile_model=True`와 함께 쓰면 "untested" warning을 냅니다.

### 6.3 r² = Gaussian 사후분산 (`rtc_guidance_r2="posterior"`, 실험용)

논문은 단위분산 prior를 가정해 `r² = 1 − a`를 씁니다. 이 변형은 원소별 사후분산으로 바꿉니다.

1. **σ² 추정** (`estimate_prior_variance`): 같은 관측(같은 `global_cond`를 `repeat_interleave`)에서 unguided chunk K개를 `no_grad`로 뽑습니다. 원소별 unbiased 분산을 계산해 `σ² ∈ (B, H, A)`로 쓰고, 아래쪽을 1e-6으로 막습니다.
   - 난수는 별도 `torch.Generator`에서 뽑습니다. seed는 `2506_07339 + 호출 횟수`이고, 카운터는 `init_rtc_processor`에서 0으로 리셋됩니다.
   - 따라서 전역 RNG를 소비하지 않고, guided chunk의 noise는 다른 변형과 같습니다.
2. **r²**: `r²_d = σ²_d (1−a) / (a σ²_d + 1−a)`
3. **가중치** λ_d는 VJP 전에 cotangent에 곱하고(`c = λ ⊙ e`), 이때 `w = 1`입니다.

   | `rtc_r2_clip` | `pigdm` | `flow` |
   |---|---|---|
   | `relative` | `λ = w_unit(a, β) · (1−a)/r²_d` | 같은 식(`w_unit`만 flow) |
   | `literal` | `λ = min(β, 1/r²_d)` | `λ = min(β, c/r²_d)/c`, `c = √((1−a)/a)` |

   clip이 걸리지 않으면 둘 다 `1/r²_d`입니다.

- **이론적 근거**: 독립 Gaussian prior `x0_d ~ N(μ_d, σ²_d)`에서는 `J_dd = √a σ² / (a σ² + 1−a)`이므로 `J/r² = √a/(1−a)`가 σ²와 무관하게 정확히 성립합니다. clip이 없으면 `Δx0_hat = W ⊙ e`, 즉 x0_hat이 목표에 정확히 도달합니다. 두 성질 모두 테스트로 확인했습니다.
- **적용 조건**: guide가 활성이고 mode가 `guidance`일 때만 계산합니다. `identity` Jacobian과도 함께 쓸 수 있습니다(`g = s · λ ⊙ e`).
- **비용**: 호출마다 B·K개 chunk의 unguided 샘플링이 한 번 추가됩니다.
- **안정성**:
  - `relative`는 보정을 최대 `(1−a)/σ²`배까지 증폭합니다. 모델의 Jacobian이 σ²에 비례할 때만 안정적입니다.
  - 무작위 초기화 U-Net에서는 발산하고, 학습된 PushT 모델에서도 발산했습니다(§11.4).

### 6.4 replacement 모드

- 매 `scheduler.step` 직후, `W ≥ 1`인 slot을 `q(x_{t_prev} | Y)`의 샘플 `√a_prev · Y + √(1−a_prev) · n`으로 덮어씁니다. 마지막 step에서는 `Y`를 그대로 씁니다.
- `t_prev`는 scheduler와 같은 방식으로 계산합니다. DDIM은 `t − T_train / N`이고, DDPM은 `scheduler.previous_timestep`입니다. noise는 같은 `generator`에서 뽑습니다.
- 덮어쓰는 대상은 `W ≥ 1`인 slot뿐입니다. `ZEROS`이면 frozen prefix `[0, d)`, `ONES`이면 overlap 전체입니다.
- VJP가 없으므로 비용은 unguided와 같습니다.

### 6.5 비활성 guide

mode가 `none`이거나 `W`가 모두 0이면(leftover 없음, `ZEROS`에서 d = 0 등) 원래 sampler와 같은 연산을 수행하므로 출력이 bit 단위로 같습니다.

## 7. 디버그와 계측

`RTCConfig.debug=True`이면 step마다 LeRobot RTC debug tracker 형식(`time = t / num_train_timesteps`; 1은 noise, 0은 data)으로 기록합니다. 같은 값이 `last_rtc_info["steps"]`에도 남습니다.

| 키 | 의미 |
|---|---|
| `timestep`, `alpha_bar` | step 정보 |
| `guidance_weight` | `w`. posterior에서는 가중치가 있는 원소의 λ 평균 |
| `err_norm`, `grad_norm` | `‖e‖`, `‖g‖` |
| `delta_out_norm`, `delta_x0_norm` | 모델 출력과 x0_hat의 변화량 |
| `corr_frac` | `⟨Δx0_hat, e⟩ / ‖e‖²`. 이 step에서 제거된 prefix 오차의 비율이며, 1이면 x0_hat이 목표에 정확히 도달한 것 |

## 8. 가정과 확인하지 못한 사항

| ID | 내용 | 상태 |
|---|---|---|
| A1 | `r² = 1 − a`는 단위분산 prior 가정. MIN_MAX 정규화된 action은 분산이 1보다 작음 | `posterior` 변형으로 검증했고, 원인이 r²가 아님을 확인(§11.4) |
| A2 | 기본 β는 flow-matching 기준값. `pigdm` clip에서는 저잡음 쪽 몇 step에만 영향 | sweep 필요. PushT에서는 sweep함 |
| A3 | DDPM 다단계에서 초반 ᾱ ≈ 1e-7이면 Jacobian이 `1/√a`로 증폭됨 | DDIM-10 사용 권장(warning) |
| A4 | `inference_delay=None`은 0으로 처리(pi0 processor는 None에서 실패함) | 설계 선택 |
| A5 | posterior r²는 대각 σ²만 사용. 시간·차원 간 상관과 다봉성은 무시 | 미검증 |
| A6 | `compile_model=True`와 VJP를 함께 쓰는 경우 | 미검증(warning) |
| A7 | `lerobot-rollout`은 leftover를 `execution_horizon` 길이로 0 패딩함. 실제 leftover가 더 짧으면 0이 target이 됨 | 기존 동작이며 pi0/smolvla도 같음. 수정하지 않음 |
| A8 | `lerobot-rollout`의 `queue_threshold` 기본값 30은 `H_eff = 15`보다 커서 쉬지 않고 재계획함 | `--inference.queue_threshold`를 `H_eff`보다 작게 지정해야 함 |

## 9. 테스트

`tests/policies/diffusion/test_diffusion_rtc.py`에 252개 케이스가 있습니다.

```bash
uv run pytest tests/policies/diffusion/test_diffusion_rtc.py -q
```

| 분류 | 확인 내용 |
|---|---|
| bit 동일성 | RTC off/disabled, leftover 없음, `ZEROS`에서 d = 0, 가중치가 전부 0일 때 기존 sampler와 출력이 같음(prediction type × scheduler × 이미지 유무 × mode) |
| parameterization 동등성 | epsilon과 sample 보정이 같은 Δx0를 주고, guided sampling 결과도 같음(DDIM/DDPM × clip_sample × clip) |
| guidance 효과 | guidance가 prefix 오차를 줄이고, β를 키워도 오차가 늘지 않으며, replacement는 frozen prefix를 정확히 맞춤 |
| 좌표와 가중치 | index mapping(`n_obs` 1–3), 과거 action 배치, `resolve_overlap`, prefix 가중치가 `RTCProcessor`와 일치, past-action 가중치 |
| LeRobot flow RTC와 대조 | `flow` clip이 `RTCProcessor.denoise_step`과 일치하고, 진짜 VJP가 논문 flow 식(Eq. 2–4)과 일치(1e-9). `identity`가 `RTCProcessor`를 재현 |
| autograd 안전성 | guided inference 후 파라미터 `.grad`가 없고, `torch.inference_mode` 안에서도 동작 |
| shape, device, batch | dtype과 device, batch 독립성, 잘못된 leftover shape에서 예외 |
| config | 구버전 config.json 로드, 저장 후 재로드 왕복, 잘못된 값과 `v_prediction` 거부 |
| queue와 기타 | `observe`/snapshot 의미, `select_action` 가드, DDPM 체크포인트를 DDIM으로 실행, 다단계 warning, processor 재생성, debug tracker |
| posterior r² | 대각 Gaussian prior에서 `J/r² = √a/(1−a)`가 정확히 성립, 무clip에서 x0_hat이 목표에 도달, unit r²는 목표에 못 미침, 가중치 공식, σ² = 1이면 unit과 동일, end-to-end(RNG 격리, 결정성, inference_mode) |
| rollout 엔진 | tick별 관측 history가 정책에 전달됨 |

마지막 회귀 실행에서 `tests/policies/diffusion`, `tests/policies/rtc`, `tests/configs`의 481개가 모두 통과했고 ruff도 통과했습니다.

## 10. 평가 도구 (`examples/rtc_diffusion/`)

| 스크립트 | 내용 | 주요 출력 |
|---|---|---|
| `eval_dataset.py` | 데이터셋 프레임 k에서 chunk A를 만들고, k+s에서 leftover `A[s:]`로 chunk B를 만듦. variant마다 같은 초기 noise 사용 | `offline_metrics.csv`, `guidance_steps.csv`, `offline_summary.md`, PNG |
| `eval_sim_delay.py` | 스레드 없이 지연을 정수 step `d`로 모사하는 결정적 closed-loop 평가. method × d × s × β를 sweep | `summary.md`/CSV |
| `diag_jacobian_r2.py` | 학습된 J(D×D 전체)와 unit/posterior r², Gaussian 해석해, 모델 자신의 ancestral 사후공분산을 step별로 비교 | `jacobian_r2_rows.csv`, `jacobian_r2_summary.md` |

**variant/method 정의**(`eval_dataset.VARIANTS`와 `eval_sim_delay.METHODS`가 같은 이름을 씀):

| 이름 | mode | schedule | Jacobian | clip | r² |
|---|---|---|---|---|---|
| `sync` (sim 전용) | 지연 동안 마지막 action 유지 | – | – | – | – |
| `naive` | none | – | – | – | – |
| `guidance_zeros` / `_exp` / `_linear` / `_ones` | guidance | 각각 | vjp | `--weight_clip` | unit |
| `guidance_zeros_id` / `guidance_exp_id` | guidance | ZEROS / EXP | identity | flow | unit |
| `guidance_exp_post` | guidance | EXP | vjp | `--weight_clip` | posterior, relative |
| `guidance_exp_post_lit` | guidance | EXP | vjp | `--weight_clip` | posterior, literal |
| `replacement` / `replacement_ones` | replacement | ZEROS / ONES | – | – | – |

`_post` 변형은 기본 목록에 없고, 직접 지정해야 실행됩니다.

**sim 타임라인**:

1. step t에 관측을 받고 inference를 시작합니다. 현재 chunk의 남은 부분을 leftover로 넘깁니다.
2. t부터 t+d−1까지는 이전 chunk를 계속 실행합니다.
3. t+d에 새 chunk가 index d부터 이어받습니다.
4. 다음 inference는 t+s에 시작합니다(s ≥ d).

`sync`는 d step 동안 정지했다가 새 chunk를 index 0부터 실행합니다. d = 0이면 sync, naive, 원래 DP 루프가 모두 같습니다.

**지표**:

| 지표 | 정의 |
|---|---|
| `frozen_rms_norm` | 새 chunk와 leftover의 `[0, d)` RMS 차이(정규화 단위). prefix를 얼마나 지켰는지 |
| `overlap_rms_norm` | 같은 RMS 차이를 overlap 전체에서 계산 |
| `switch_step_norm` (offline), `switch_disc` (sim) | chunk 전환 시점의 action 점프 크기. sim은 env 단위 |
| `max_d2_norm`, `jerk_mean`, `jerk_switch` | chunk 내부와 궤적의 2차·3차 차분 |
| `gt_rmse_env` | 데이터셋 GT action과의 RMSE(env 단위) |
| `corr_frac_med` | step별 `corr_frac`의 중앙값 |
| `success_rate`, `max_reward`, `starved_steps` | 과제 성능과 chunk 소진 횟수 |
| `latency_ms_*` | CUDA 동기화한 추론 시간. guided와 unguided 호출을 분리해 측정 |

## 11. PushT 측정 결과

설정은 다음과 같습니다.

- 모델: `lerobot/diffusion_pusht`. `migrate_policy_normalization.py`로 processor 형식으로 변환했습니다.
- 모델 구성: DDPM-100(squaredcos_cap_v2)으로 학습된 모델을 DDIM-10으로 추론합니다. epsilon prediction, `clip_sample=True`, `H = 16`, `n_obs = 2`, `H_eff = 15`, `A = 2`입니다.
- GPU: RTX 5070 Ti
- sim: 20 에피소드, `replan_every = 8`, B = 10. success 표준오차는 약 0.11이므로 **성공률 차이에 대한 결론은 내리지 않습니다.**

### 11.1 지연 sweep (sim, β = 10; VJP는 `pigdm`, `_id`는 `flow`)

셀 값은 `switch_disc / frozen_rms_norm (success)`입니다. d = 0에서는 모든 method가 원래 DP와 같으며, switch_disc 9.89, success 0.80입니다.

| d | sync | naive | guidance_exp (VJP) | guidance_exp_id | replacement |
|---|---|---|---|---|---|
| 1 | 9.84 / – (0.60) | 11.55 / 0.038 (0.80) | 11.70 / 0.026 (0.55) | **9.09 / 0.016** (0.75) | 11.87 / 0.000 (0.65) |
| 2 | 9.91 / – (0.70) | 15.11 / 0.044 (0.65) | 14.11 / 0.044 (0.70) | **10.78 / 0.018** (0.65) | 14.69 / 0.000 (0.65) |
| 4 | 9.93 / – (0.55) | 18.35 / 0.060 (0.70) | 20.18 / 0.057 (0.55) | **14.28 / 0.023** (0.60) | 18.31 / 0.000 (0.60) |
| 6 | 9.89 / – (0.40) | 30.00 / 0.087 (0.45) | 27.86 / 0.093 (0.60) | **21.67 / 0.032** (0.60) | 25.85 / 0.000 (0.40) |

- sync는 전환 점프가 작지만, 지연 동안 정지하므로 d가 커질수록 성공률이 떨어지는 경향을 보입니다.
- replacement는 frozen prefix를 정확히 지키지만 바로 다음 slot과의 점프는 줄이지 못합니다.
- 논문식 VJP guidance는 naive와 거의 구분되지 않습니다.

### 11.2 β와 clip (sim, `flow` clip, switch_disc / frozen RMS)

| d | naive | VJP β=10 | VJP β=100 | identity β=10 | identity β=100 |
|---|---|---|---|---|---|
| 2 | 15.1 / 0.044 | 14.2 / 0.033 | 11.5 / 0.023 | 10.8 / 0.018 | **8.9 / 0.001** |
| 4 | 18.4 / 0.060 | 16.5 / 0.047 | 14.2 / 0.033 | 14.3 / 0.023 | **11.1 / 0.001** |
| 6 | 30.0 / 0.087 | 28.3 / 0.082 | 22.0 / 0.050 | 21.7 / 0.032 | **19.9 / 0.001** |

### 11.3 오프라인과 지연 시간

| variant | frozen RMS (d=3, β=10; VJP는 `pigdm`, `_id`는 `flow`) | step별 `corr_frac` 중앙값 (β=100, `flow`) | latency B=1 | latency B=10 |
|---|---|---|---|---|
| naive | 0.016 | – | 65 ms | 80 ms |
| guidance_exp (VJP) | 0.015 | 0.004 | 100 ms | 232–240 ms |
| guidance_exp_id | 0.009 | 0.999 | 69 ms | 80–83 ms |
| replacement | 0.000 | – | 66 ms | 80 ms |
| guidance_exp_post_lit | – | 0.167 | 183 ms | ~410 ms |

### 11.4 r² 가설 검증 (`diag_jacobian_r2.py`, 24 프레임, K = 64, M = 64)

가설은 "VJP guidance가 약한 원인이 단위분산 가정(r² = 1−a)"이라는 것이었습니다. **결과는 기각입니다.**

- 조건부 action 분산은 매우 작습니다. σ²의 중앙값이 3.4e-5(σ ≈ 0.006)이고, 단위 r²는 마지막 step을 빼면 10³–10⁶배 과대입니다.
- 보정한 r²는 정확합니다. 모델 자신의 ancestral 사후분산과의 비가 모든 step에서 0.89–1.05입니다.
- 학습된 J는 이와 맞지 않습니다.
  - `(J/r²_post) ÷ (√a/(1−a))`의 중앙값이 step 0–7에서 140, 33, −27, −7.2, −0.55, 17.7, 5.5, 3.9입니다. p10–p90 구간은 부호가 섞여 있습니다.
  - 비대칭도는 0.48–0.66입니다. 정확한 Tweedie Jacobian은 대칭이어야 합니다.
  - fp64에서도 같은 값이 나오므로 수치 오차가 아닙니다.
- 원인은 epsilon parameterization에서의 상쇄로 봅니다.
  - `√(1−a)·∂ε/∂x = I − √a·J`이므로 J는 거의 항등인 사상의 잔차입니다.
  - 네트워크의 상대 정확도는 약 1%인데, 참값 잔차는 1e-6–3e-3입니다. 그래서 J는 거의 근사 오차입니다.
  - `1/r²_post ~ 1/σ²`가 이 오차를 증폭하므로 `relative`는 모든 β에서 발산했습니다(32프레임 중 finite 0개). `literal`은 β가 크면 발산하고, β가 작으면 unit r²와 거의 같습니다.
- sim에서 `posterior_lit` β=100은 d가 커질수록 불안정했습니다. d=6에서 frozen RMS 0.274, jerk 34.9였습니다.

## 12. 열린 설계 이슈와 후속 후보

1. **해석적 Gaussian Jacobian 변형 (`jacobian="gaussian"`, 미구현)**: J도 같은 Gaussian 모델의 `J_d = √a σ²/(a σ² + 1−a)`를 쓰면 σ²가 약분되어 `g = √a/(1−a) · W ⊙ e`, 즉 `Δx0_hat = W ⊙ e`가 됩니다. VJP도 K개 샘플링도 필요 없습니다.
   - 현재 `identity`는 이 식에 `s/√a`를 곱한 것과 같습니다. 측정한 corr_frac 6.33, 2.75, 1.71, …이 `s/√a`와 일치합니다.
   - 이 변형은 고잡음 step의 과보정을 없앤 원리적 수정판이 됩니다.
   - σ²로 full Σ를 추정하면 prefix 보정이 제약 없는 인접 slot으로 전파되는 확장도 가능합니다.
2. **sample-prediction 체크포인트로 상쇄 가설 확인**: §11.4의 메커니즘이 맞다면 sample prediction에서는 VJP가 정확해야 합니다. 재학습이 필요합니다.
3. **성공률 비교**: config당 100 에피소드 이상이 필요합니다.
4. **`lerobot-rollout` leftover 0 패딩(A7)**: 기존 동작이며, 여러 정책에 공통으로 영향을 줍니다.
5. **`compile_model`과 VJP, 실제 로봇 rollout**: 검증하지 않았습니다.
6. **재현용 체크포인트**: 변환한 체크포인트는 저장소 밖에 있습니다. 다음 명령으로 다시 만듭니다.

   ```bash
   python src/lerobot/processor/migrate_policy_normalization.py \
     --pretrained-path lerobot/diffusion_pusht --output-dir <out_dir>
   ```

## 부록: 재현 명령

```bash
CKPT=<out_dir>
COMMON="--policy.path=$CKPT --policy.noise_scheduler_type=DDIM --policy.num_inference_steps=10"

# 오프라인 (variant 비교, β와 clip 지정)
python examples/rtc_diffusion/eval_dataset.py $COMMON --dataset.repo_id=lerobot/pusht \
  --inference_delay=3 --replan_every=8 --num_samples=32 --weight_clip=flow --rtc.max_guidance_weight=100 \
  --variants='[naive,guidance_exp,guidance_exp_post,guidance_exp_post_lit,guidance_exp_id]'

# 지연 sweep (sim)
python examples/rtc_diffusion/eval_sim_delay.py $COMMON --env.type=pusht --n_episodes=20 --batch_size=10 \
  --delays='[0,1,2,3,4,6]' --replan_every='[8]' --betas='[10.0]'

# β와 r² 비교 (sim)
python examples/rtc_diffusion/eval_sim_delay.py $COMMON --env.type=pusht --n_episodes=20 --batch_size=10 \
  --weight_clip=flow --methods='[guidance_exp,guidance_exp_post_lit,guidance_exp_id]' \
  --delays='[2,4,6]' --replan_every='[8]' --betas='[10.0,100.0]'

# Jacobian / r² 진단
python examples/rtc_diffusion/diag_jacobian_r2.py $COMMON --dataset.repo_id=lerobot/pusht \
  --num_frames=24 --num_var_samples=64 --num_posterior_samples=64
```
