"""Smoke test: OpenArm IDM model + dataloader wiring before launching training."""
import os
from hydra.utils import instantiate
from omegaconf import OmegaConf

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config_idm import DATA_CONFIG_MAP

DATA_CFG = "openarm_ego_ae20"
TRAIN = os.path.expanduser("~/data/openarm-idm/train")

print("=== 1. data config ===")
dc = DATA_CONFIG_MAP[DATA_CFG]
mc = dc.modality_config()
print("modality keys:", {k: v.modality_keys for k, v in mc.items()})
print("video delta (obs indices):", mc["video"].delta_indices,
      "| action horizon:", len(mc["action"].delta_indices))

print("\n=== 2. dataset (computes stats.json from parquet on first load) ===")
ds = LeRobotSingleDataset(
    dataset_path=TRAIN,
    modality_configs=mc,
    transforms=dc.transform(),
    embodiment_tag=EmbodimentTag("new_embodiment"),
    video_backend="decord",
)
print("num steps:", len(ds), "| num trajectories:", len(ds.trajectory_ids))
sample = ds[0]
for k, v in sample.items():
    shape = getattr(v, "shape", None)
    print(f"  {k}: {type(v).__name__} shape={shape}")

print("\n=== 3. model (instantiates SigLIP2 + DiT, action_dim should be 28) ===")
model = instantiate(OmegaConf.load("IDM_dump/configs/openarm_ae20.yaml"))
print("model action_dim:", model.config.action_dim,
      "| action_horizon:", model.config.action_horizon)
nparams = sum(p.numel() for p in model.parameters()) / 1e6
print(f"total params: {nparams:.1f}M")
print("\nSMOKE TEST PASSED")
