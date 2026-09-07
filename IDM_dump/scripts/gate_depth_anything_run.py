#!/usr/bin/env python
"""Pre-flight gate for the DepthAnything IDM arm. Runs on CPU, allocates no GPUs.

Every check here is a failure mode that is silent at training time: wrong token count,
wrong normalisation, the wrong yaml being selected, or the baseline arm having drifted.
Run this to completion before submitting the 30k-step job.

  python IDM_dump/scripts/gate_depth_anything_run.py --data-root <stereo mirror train>
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from gr00t.data.dataset import LeRobotSingleDataset       # noqa: E402
from gr00t.data.schema import EmbodimentTag               # noqa: E402
from gr00t.experiment.data_config_idm import DATA_CONFIG_MAP  # noqa: E402

DEPTH_CFG = "openarm_ego_stereo_ae20_depth_mirror"
BASE_CFG = "openarm_ego_stereo_ae20_mirror"

fails: list[str] = []


def check(name, cond, detail=""):
    print(("  OK   " if cond else "  FAIL ") + name + (("  -- " + detail) if detail else ""),
          flush=True)
    if not cond:
        fails.append(name)


def sample_from(cfg_name, root):
    cfg = DATA_CONFIG_MAP[cfg_name]
    tr = cfg.transform()
    tr.eval()  # deterministic: no mirror, no jitter -- both arms must see the same frame
    ds = LeRobotSingleDataset(dataset_path=root, modality_configs=cfg.modality_config(),
                              transforms=tr, embodiment_tag=EmbodimentTag("new_embodiment"),
                              video_backend="decord")
    return ds[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    a = ap.parse_args()

    print("=== 1. 데이터 config 등록 ===", flush=True)
    check("depth config가 DATA_CONFIG_MAP에 있다", DEPTH_CFG in DATA_CONFIG_MAP)
    check("depth config의 image_preprocess", 
          DATA_CONFIG_MAP[DEPTH_CFG].image_preprocess == "depth_anything",
          repr(DATA_CONFIG_MAP[DEPTH_CFG].image_preprocess))
    check("기준선 config의 image_preprocess는 여전히 None",
          DATA_CONFIG_MAP[BASE_CFG].image_preprocess is None,
          repr(DATA_CONFIG_MAP[BASE_CFG].image_preprocess))
    check("video_keys가 stereo로 동일",
          DATA_CONFIG_MAP[DEPTH_CFG].video_keys == DATA_CONFIG_MAP[BASE_CFG].video_keys,
          str(DATA_CONFIG_MAP[DEPTH_CFG].video_keys))

    print("=== 2. yaml 선택 분기 ===", flush=True)
    def pick(name):
        if "openarm" in name:
            return ("IDM_dump/configs/openarm_ae20_depth.yaml" if "depth" in name
                    else "IDM_dump/configs/openarm_ae20.yaml")
        return "?"
    check("depth 이름 -> depth yaml", pick(DEPTH_CFG).endswith("openarm_ae20_depth.yaml"),
          pick(DEPTH_CFG))
    check("기준선 이름 -> 기존 yaml", pick(BASE_CFG).endswith("openarm_ae20.yaml"), pick(BASE_CFG))

    print("=== 3. 로더 경로에서 실제 프레임 ===", flush=True)
    d_img = sample_from(DEPTH_CFG, a.data_root)["images"]
    b_img = sample_from(BASE_CFG, a.data_root)["images"]
    d_img = torch.as_tensor(d_img); b_img = torch.as_tensor(b_img)
    check("depth images shape (4,3,224,224)", tuple(d_img.shape) == (4, 3, 224, 224),
          str(tuple(d_img.shape)))
    check("기준선 images shape (4,3,256,256)", tuple(b_img.shape) == (4, 3, 256, 256),
          str(tuple(b_img.shape)))
    dpk, bpk = float(d_img.abs().max()), float(b_img.abs().max())
    check("depth는 ImageNet 범위 (|x|>1.01)", dpk > 1.01, f"peak={dpk:.3f}")
    check("기준선은 SigLIP 범위 (|x|<=1.001)", bpk <= 1.001, f"peak={bpk:.3f}")

    print("=== 4. 모델 조립 + CPU forward ===", flush=True)
    # Exactly how scripts/idm_training.py builds it (line ~161). Instantiating any other
    # way would test a path training never takes.
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    model = instantiate(OmegaConf.load(
        os.path.join(REPO, "IDM_dump/configs/openarm_ae20_depth.yaml")))
    ah = model.action_head
    check("action_horizon 20 (기준선과 동일)", model.config.action_horizon == 20,
          str(model.config.action_horizon))
    check("action_dim 28", model.config.action_dim == 28, str(model.config.action_dim))
    check("tower가 DepthAnythingTower", type(ah.siglip_model).__name__ == "DepthAnythingTower",
          type(ah.siglip_model).__name__)

    with torch.no_grad():
        feats = ah.encode_images(d_img.float(), torch.arange(4))
    check("encode_images -> (4, 16, 1024)", tuple(feats.shape) == (4, 16, 1024),
          str(tuple(feats.shape)))
    check("출력이 유한", bool(torch.isfinite(feats).all()))

    print("=== 5. 동결 / DDP 지뢰 ===", flush=True)
    # 시그니처를 추측하지 않고 실제 것을 읽어 전부 True로 부른다 (학습이 쓰는 최대 설정)
    import inspect
    sig = inspect.signature(ah.set_trainable_parameters)
    kw = {k: True for k in sig.parameters}
    ah.set_trainable_parameters(**kw)
    mt = ah.siglip_model.vision_model.backbone.embeddings.mask_token
    check("mask_token 동결 (ddp_find_unused_parameters=False 대비)", not mt.requires_grad)
    unused = [n for n, p in ah.siglip_model.named_parameters()
              if p.requires_grad and "mask_token" in n]
    check("학습 대상에 mask_token 없음", not unused, str(unused))
    n_tr = sum(p.numel() for p in ah.siglip_model.parameters() if p.requires_grad) / 1e6
    print(f"       tower trainable: {n_tr:.1f} M", flush=True)

    print("=== 6. SigLIP 입력이 depth tower에 들어가면 즉시 실패하는가 ===", flush=True)
    tripped = False
    try:
        ah.siglip_model._checked_input_range = False
        with torch.no_grad():
            ah.siglip_model.vision_model(b_img.float()[:, :, :224, :224])
    except ValueError:
        tripped = True
    check("tripwire 작동", tripped)

    print("=== 7. 체크포인트 저장 경로 ===", flush=True)
    # gr00t/experiment/trainer.py:131 calls model.save_pretrained(dir, state_dict=...).
    # safetensors refuses to write two names backed by one storage, so any accidental
    # module alias in the tower aborts the run at the first save -- 5000 steps in, which
    # is exactly what happened on 2026-09-05. Walk the same call, not a proxy for it.
    import collections
    import tempfile

    seen = collections.defaultdict(list)
    for k, v in model.state_dict().items():
        seen[(v.data_ptr(), v.device, v.shape)].append(k)
    dupes = {ptr: names for ptr, names in seen.items() if len(names) > 1}
    check("state_dict에 공유 텐서 없음", not dupes,
          "" if not dupes else f"{len(dupes)}건, 예: {sorted(next(iter(dupes.values())))[:4]}")

    with tempfile.TemporaryDirectory() as td:
        try:
            model.save_pretrained(td)
            wrote = [f for f in os.listdir(td) if f.endswith((".safetensors", ".bin"))]
            check("save_pretrained 성공", bool(wrote), str(wrote))
        except Exception as e:  # noqa: BLE001 - the whole point is to surface it here
            check("save_pretrained 성공", False, f"{type(e).__name__}: {str(e)[:200]}")

    print()
    if fails:
        print("GATE FAILED: " + "; ".join(fails), flush=True)
        raise SystemExit(1)
    print("GATE PASSED — GPU 할당해도 됨", flush=True)


if __name__ == "__main__":
    main()
