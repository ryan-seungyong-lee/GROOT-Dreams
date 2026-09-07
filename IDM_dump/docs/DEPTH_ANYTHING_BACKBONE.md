# DepthAnything-V2 vision tower for the OpenArm IDM

The IDM is video-only: it reads frames at `t` and `t+H` and predicts the 20-step, 28-D
action chunk that connects them. Its vision encoder lives **inside the action head**
(`self.siglip_model`), not in the backbone slot — `IdentityBackbone` is a dead
pass-through. So swapping the encoder means swapping that attribute, not the backbone.

This documents the swap from SigLIP2-large to **DepthAnything-V2-Large**, run as a
single-variable A/B, and how to train and evaluate either arm.

Motivation: 1X's world-model post reports an IDM built from a Depth Anything backbone
feeding a separate flow-matching head. Their setup (W=8, 400 h of data) is not
reproducible here — we have 3.75 h — so the question is narrower: **does a
depth-pretrained representation beat a contrastive-pretrained one on our data?**

## Result

Same dataset, same `action_horizon=20`, same global batch 128, same 30 000 steps, same
mirror augmentation, same stereo pair. Only the vision tower differs. Evaluated on the
same 34 test episodes (1013 query chunks, 20 260 step samples) by one script in one job.

| | SigLIP2-large | **DepthAnything-V2-L** | |
|---|---|---|---|
| overall MAE | 0.0359 | **0.0336** | **−6.4 %** |
| overall RMSE | 0.0754 | 0.0718 | −4.8 % |

Per group (radians), all five improve:

| group | SigLIP | Depth | |
|---|---|---|---|
| left_arm | 0.0230 | 0.0215 | −6.5 % |
| right_arm | 0.0330 | 0.0303 | −8.2 % |
| left_hand | 0.0369 | 0.0354 | −4.1 % |
| right_hand | 0.0651 | 0.0608 | −6.6 % |
| neck | 0.0000 | 0.0000 | — |

Error decomposition — most of the gain is in the **pose** term, which is the part that
reads absolute joint angles out of pixels. That is the term a depth-pretrained
representation should help:

| | SigLIP | Depth |
|---|---|---|
| pose (absolute) | 0.0360 | 0.0338 |
| motion (within chunk) | 0.0169 | 0.0163 |

Uniform across the horizon: t+1 −6.0 %, t+5 −6.9 %, t+10 −6.4 %, t+15 −5.9 %,
t+20 −5.5 %.

### Caveat worth stating plainly

**Both arms are still worse than the static baseline.** Repeating `action[t]` for 20
steps gives MAE **0.0281**, against 0.0336 (depth) and 0.0359 (SigLIP). Only
`right_arm` beats it (0.0303 vs 0.0389). The backbone comparison is unaffected — the
static baseline is identical for both arms — but the IDM does not yet beat "hold still"
overall, and the 6.4 % is an improvement on top of that.

## Why the swap is dimension-compatible

| | SigLIP2-large | DepthAnything-V2-Large |
|---|---|---|
| hidden | 1024 | **1024** (identical) |
| patch | 16 | 14 |
| input | 256×256 | **224×224** |
| patches | 16×16 = 256 | **16×16 = 256** (identical) |
| tower params | 316.0 M | ~335 M |

`mm_projector` (`mlp_doubledownsample`) infers `h = w = int(seq_len ** 0.5)` and pools
2×2 twice, so it must receive exactly 256 patches to emit the 16 tokens
`num_visual_tokens_per_frame` expects. 224 input at patch 14 gives 256 → 64 → 16, the
same as now, which is why `vision_projector`, `mm_projector`, `masked_scatter` and
`max_sequence_length=112` are all untouched. Everything outside the tower (DiT, action
encoder/decoder, VL self-attention — 321 M params) is identical between the arms.

252 input would **not** work: 18×18 = 324 patches → 81 tokens, and
`num_visual_tokens_per_frame` would have to change with it.

## Traps that fail silently

These are the ones that produce no error, just worse numbers.

1. **Normalisation differs.** SigLIP uses mean/std 0.5 → `[-1, 1]`; DepthAnything uses
   ImageNet stats → `[-2.118, 2.640]`. Reusing the SigLIP processor mis-normalises
   without raising. `DepthAnythingTower.check_input_range()` therefore raises on the
   first forward if the input looks like SigLIP range.
2. **DepthAnything's processor defaults to size 518.** Left alone it upsamples 224 to
   518 → 37×37 = 1369 patches, and `int(1369 ** 0.5) = 37` breaks the 2×2 pooling.
   `preprocess_image()` in `transforms_idm.py` forces 224 instead of using the
   processor.
3. **DINOv2 prepends a CLS token.** `feature_maps[-1]` is `(N, 1+256, 1024)`. It must be
   sliced `[:, 1:]`. Unsliced, 257 passes `int(257 ** 0.5) == 16` and the wrong tokens
   flow on quietly.
4. **Preprocessing is configured in two places** — the model in
   `IDM_dump/configs/openarm_ae20*.yaml`, the pixels in the `GR00TIDMTransform` field
   `image_preprocess`. Change one and the model is DepthAnything while the input is
   SigLIP-shaped.
5. **Eval builds its own transforms.** `eval_idm_openarm.py:build_eval_transforms()`
   constructs `GR00TIDMTransform` directly, so it reads the tower out of the checkpoint
   config (`infer_image_preprocess()`) rather than trusting a flag — a manual flag is
   something you forget, and forgetting it is silent.
6. **`mask_token` never gets a gradient.** `Dinov2Embeddings.mask_token` is untouched by
   any forward path. Training runs with `ddp_find_unused_parameters=False`
   (`scripts/idm_training.py`), which aborts the first DDP step on such a parameter —
   invisible on 1 GPU. `freeze_unused_parameters()` freezes it.
7. **safetensors refuses shared tensors.** An earlier version of the tower registered
   `backbone.encoder` under a second name to expose `.layers`, which made every encoder
   parameter appear four times in `state_dict`; `save_pretrained` then raised — at the
   *first checkpoint*, 5000 steps and 72 minutes in. The alias is now a plain non-Module
   view (`_EncoderView`), and the gate calls `save_pretrained` for real.

## Design: an adapter, so the baseline arm stays byte-identical

The action head touches the tower in exactly two places: the first line of
`encode_images()` (`self.siglip_model.vision_model(images)["last_hidden_state"]`) and
four attributes in `set_trainable_parameters()` (`logit_scale`, `logit_bias`,
`vision_model.encoder.layers[11]`, `vision_model.head`). `DepthAnythingTower` presents that
same shape, so the shared code needed no edits beyond one `hasattr` dispatch for
`freeze_unused_parameters()`. The `image_preprocess` field defaults to `None`, which is
the existing SigLIP path.

That matters for the A/B: **the baseline arm runs the same bytes it ran before**, so a
difference in MAE cannot come from a refactor.

One deliberate asymmetry: the baseline freezes `encoder.layers[11]` — a middle layer of
a 24-layer model, apparently left over from a 12-layer era. DINOv2-Large is also 24
layers, so the depth arm freezes the same index. "Fixing" it on one side only would make
the fix part of the measured difference.

## Files

| path | what |
|---|---|
| `gr00t/model/action_head/depth_anything_tower.py` | the tower: SigLIP-shaped adapter over `Dinov2Backbone`, CLS slicing, input-range tripwire, `freeze_unused_parameters()` |
| `gr00t/model/transforms_idm.py` | `preprocess_image()` (224 + ImageNet stats) and the `image_preprocess` field; `None` keeps SigLIP behaviour |
| `gr00t/experiment/data_config_idm.py` | `openarm_ego_stereo_ae20_depth_mirror` |
| `IDM_dump/configs/openarm_ae20_depth.yaml` | model config; differs from `openarm_ae20.yaml` only in `siglip_model_cfg` and `siglip_hidden_size` |
| `scripts/idm_training.py` | picks the depth yaml when the data-config name contains `depth` |
| `IDM_dump/scripts/gate_depth_anything_run.py` | pre-flight gate, CPU only, no GPUs |
| `IDM_dump/scripts/eval_idm_openarm.py` | `infer_image_preprocess()` reads the tower from the checkpoint |

## Run it

### Gate first (CPU, ~2 min, no GPU)

Every check in the gate is a failure mode that is silent during training. It is much
cheaper than finding out at step 30 000.

```bash
python IDM_dump/scripts/gate_depth_anything_run.py \
  --data-root <dataset>/train
```

It asserts, among 21 checks: token count `(4, 16, 1024)` out of `encode_images`, depth
input at `(4, 3, 224, 224)` in ImageNet range, the baseline still at `(4, 3, 256, 256)`
in SigLIP range, `action_horizon == 20`, `mask_token` frozen, no shared tensors in
`state_dict`, and a real `save_pretrained`.

### Train

```bash
export IS_TORCHRUN=1
torchrun --standalone --nproc_per_node=4 --nnodes=1 \
  scripts/idm_training.py \
  --dataset-path <dataset>/train \
  --data-config openarm_ego_stereo_ae20_depth_mirror \
  --output-dir  <run-dir> \
  --num-gpus 4 --batch-size 32 \
  --max-steps 30000 --save-steps 5000
```

`gradient_accumulation_steps` is hardcoded to 1, so per-device batch is the only lever
on global batch size: 32 × 4 = 128 matches the baseline. Reference cost: 7 h 11 min on
4×H200 (`train_samples_per_second` 148.4, 14.2 epochs).

For the SigLIP baseline, use `--data-config openarm_ego_stereo_ae20_mirror`; the
training script then picks `openarm_ae20.yaml` on its own.

### Evaluate

No flag selects the preprocessing — it is read from the checkpoint, so the same command
scores either arm:

```bash
python IDM_dump/scripts/eval_idm_openarm.py \
  --checkpoint  <run-dir> \
  --test-dir    <dataset>/test \
  --train-dir   <dataset>/train \
  --output-dir  <out>/depth \
  --max-episodes 34 \
  --video-keys video.camera_ego_left video.camera_ego_right
```

`--video-keys` order is part of the model's interface: the view embedding is indexed by
position, so left then right, matching the data config's `video_keys`. Train
normalisation stats are used for the test set (copied in by the script). Output is
`eval_report.json` plus overlay PNGs.

## Checkpoint

`RyanL22/openarm-idm-depthanything-v2-large` on the Hugging Face Hub — the 30 000-step
model evaluated above.

## Deferred

- No-mirror ablation.
- `action_horizon` down to 5–10 (kept at 20 here so the existing baseline stays usable
  as the control).
- The 224-vs-256 resolution difference is currently paid as wasted decode: frames are
  decoded at 256 and resized to 224.
