# IDM_dump — OpenArm inverse dynamics model

An inverse dynamics model (IDM) for a bimanual OpenArm robot with Inspire RH56F1 hands.
It is **video-only**: given frames at `t` and `t+20` from a stereo egocentric camera
pair, it predicts the 20-step, 28-D action chunk that connects them
(neck 2 + left arm 7 + right arm 7 + left hand 6 + right hand 6). Its use is labelling
generated or unlabelled robot video with actions.

The vision encoder lives **inside the action head** (`self.siglip_model`), not in the
backbone slot — `IdentityBackbone` is a dead pass-through. Two encoders are supported:

| data config | model config | vision tower |
|---|---|---|
| `openarm_ego_stereo_ae20_mirror` | `configs/openarm_ae20.yaml` | SigLIP2-large, 256×256 |
| `openarm_ego_stereo_ae20_depth_mirror` | `configs/openarm_ae20_depth.yaml` | DepthAnything-V2-Large, 224×224 |

**Start here: [`docs/DEPTH_ANYTHING_BACKBONE.md`](docs/DEPTH_ANYTHING_BACKBONE.md)** — the
A/B result, the design, the seven silent failure modes, and the exact commands to gate,
train and evaluate either arm.

Trained checkpoint: [`RyanL22/openarm-idm-depthanything-v2-large`](https://huggingface.co/RyanL22/openarm-idm-depthanything-v2-large).

## Contents

| path | what |
|---|---|
| `configs/openarm_ae20.yaml` | SigLIP baseline model config (horizon 20, action_dim 28) |
| `configs/openarm_ae20_depth.yaml` | same, with the DepthAnything tower |
| `configs/state_model_teleop20.json` | fitted action → `observation.state` constants |
| `scripts/gate_depth_anything_run.py` | pre-flight gate: 21 checks, CPU only, no GPUs. **Run before spending GPU hours** |
| `scripts/verify_depth_anything_tower.py` | narrower tower-contract check (token count, CLS, normalisation, `mask_token`) |
| `scripts/eval_idm_openarm.py` | test-split eval: per-group MAE/RMSE, pose/motion split, static baseline, overlay plots |
| `scripts/smoke_test_openarm.py` | model + dataloader wiring check before launching training |
| `scripts/prepare_openarm_idm_v3v4_10hz.py` | build the merged teleop v3+v4 IDM corpus |
| `scripts/prepare_idm_teleop20_merged.py` | resample 30 Hz teleop to 20 Hz and merge into the stereo corpus |
| `scripts/idm_chunk_fusion.py` | chunk-averaging and smoothing for IDM labels (see `docs/IDM_ACTION_LABELING.md`) |
| `scripts/fit_state_model.py` | refit the action → state model on a new corpus |

Scripts here take explicit paths; the dataset roots and cluster job wrappers used to
produce our runs are not included, since they are specific to our storage layout.
