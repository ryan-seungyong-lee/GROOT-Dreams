"""Depth Anything V2 vision tower, shaped like the SigLIP one the action head expects.

Why an adapter instead of editing the action head
-------------------------------------------------
`FlowMatchingActionHeadIDM` touches its vision tower in exactly five places, and all of them
are shallow attribute access:

    :195  del self.siglip_model.text_model
    :257  self.siglip_model.logit_scale.requires_grad = False
    :258  self.siglip_model.logit_bias.requires_grad = False
    :259  self.siglip_model.vision_model.encoder.layers[11]
    :260  self.siglip_model.vision_model.head
    :323  self.siglip_model.vision_model(images)["last_hidden_state"]

Presenting the same surface means the SigLIP arm's code path is not modified at all, so the
existing checkpoint remains the exact baseline for the A/B. Editing `encode_images` with an
`if` would work too, but any change to shared code risks perturbing the arm we are measuring
against.

Shape contract (must match SigLIP2-large at 256x256 or the projector breaks)
---------------------------------------------------------------------------
`mm_projector` is `mlp_doubledownsample`: it infers `h = w = int(seq_len ** 0.5)` and pools
2x2 twice, so it needs a square patch grid that survives two halvings and lands on exactly
`num_visual_tokens_per_frame = 16`.

    SigLIP2-large   256x256, patch 16  ->  16x16 = 256 patches -> 64 -> 16 tokens, dim 1024
    DepthAnything   224x224, patch 14  ->  16x16 = 256 patches -> 64 -> 16 tokens, dim 1024

224 is what the data transform already produces (`data_config_idm.py` VideoResize 224), and
it is 16*14, so the grid is exact rather than interpolated. The SigLIP processor previously
upsampled that 224 back to 256; feeding 224 straight through drops a lossy step.

DINOv2 prepends a CLS token, so the backbone returns 257 tokens and this module slices it
off. Leaving it in would give `int(257 ** 0.5) == 16` — the projector would accept it
silently and mix a class token into the patch grid.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# The frozen-layer index the action head hardcodes. See `_FrozenCompat` below.
_HARDCODED_FROZEN_LAYER = 11


class _Head(nn.Module):
    """Stand-in for `SiglipVisionTransformer.head` (attention pooling).

    Depth Anything has no pooling head — the action head only ever calls
    `.head.parameters()` to freeze it, and an empty parameter list is the honest answer.
    """

    def forward(self, *args, **kwargs):  # pragma: no cover - never called
        raise NotImplementedError("Depth Anything tower has no pooling head")


class _EncoderView:
    """Plain (non-Module) view over the DINOv2 encoder exposing `.layers`.

    The action head freezes `vision_model.encoder.layers[11]`; DINOv2 names that list
    `.layer`. The obvious fix -- assigning `self.encoder = backbone.encoder` and
    `encoder.layers = encoder.layer` -- registers the same parameters under four names
    (`vision_model.{,backbone.}encoder.{layer,layers}.*`). Training is unaffected, but
    `save_pretrained` refuses to write shared tensors, so the run dies at the FIRST
    checkpoint save and not before: 2026-09-05, 5000 steps and 72 minutes in.

    Keeping this a non-Module that is never assigned to a Module attribute means nothing
    extra enters the module tree, and indexing still yields the real submodules, so
    freezing reaches the real parameters.
    """

    def __init__(self, encoder: nn.Module):
        object.__setattr__(self, "_encoder", encoder)

    @property
    def layers(self):
        return self._encoder.layer

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_encoder"), name)


class _VisionModel(nn.Module):
    """Exposes `.encoder.layers`, `.head`, and a `__call__` returning `last_hidden_state`."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = _Head()
        self._owner = None  # set by DepthAnythingTower; not a submodule (avoid a cycle)

    @property
    def encoder(self):
        # A property, not an attribute: see _EncoderView. Every encoder parameter must
        # appear in state_dict exactly once, under `backbone.encoder.layer.*`.
        return _EncoderView(self.backbone.encoder)

    def forward(self, pixel_values: torch.Tensor) -> dict:
        if self._owner is not None:
            self._owner.check_input_range(pixel_values)
        out = self.backbone(pixel_values)
        # Dinov2Backbone is configured with reshape_hidden_states=False and
        # apply_layernorm=True, so feature_maps[-1] is (N, 1 + P, D) post-layernorm.
        hidden = out.feature_maps[-1]
        hidden = hidden[:, 1:]  # drop CLS -> (N, P, D)
        return {"last_hidden_state": hidden}


class DepthAnythingTower(nn.Module):
    """SigLIP-shaped wrapper around the Depth Anything V2 DINOv2 backbone."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.vision_model = _VisionModel(backbone)
        # `del self.siglip_model.text_model` runs unconditionally in the action head.
        self.text_model = nn.Identity()
        # Freezing targets that exist only on SigLIP. Kept as real (tiny) parameters so
        # `.requires_grad = False` works without a special case.
        self.logit_scale = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.logit_bias = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.freeze_unused_parameters()
        self._checked_input_range = False
        object.__setattr__(self.vision_model, "_owner", self)

    def freeze_unused_parameters(self) -> None:
        """Detach parameters that never receive a gradient.

        `Dinov2Embeddings.mask_token` is only read when `bool_masked_pos` is passed, and
        `Dinov2Backbone.forward` never passes it. Training runs with
        `ddp_find_unused_parameters=False` (scripts/idm_training.py:199), so leaving it
        trainable makes DDP raise "Expected to have finished reduction in the prior
        iteration" on the first multi-GPU step. This is invisible on one GPU — it is the
        structural twin of SigLIP's unused `logit_scale`/`logit_bias`.
        """
        emb = self.vision_model.backbone.embeddings
        if hasattr(emb, "mask_token"):
            emb.mask_token.requires_grad = False

    def check_input_range(self, pixel_values: torch.Tensor) -> None:
        """Trip on SigLIP-normalised input reaching the depth tower.

        ImageNet normalisation maps [0,1] to [-2.118, 2.640]; SigLIP's mean/std of 0.5 maps
        it to exactly [-1, 1]. Any real frame has near-0 or near-255 pixels, so a max
        magnitude at or below 1 means the SigLIP preprocessing path fed this tower — the
        train/eval skew that would otherwise produce plausible-looking but wrong numbers.
        Checked once; the flag costs nothing after that.
        """
        if self._checked_input_range:
            return
        self._checked_input_range = True
        peak = float(pixel_values.abs().max())
        if peak <= 1.01:
            raise ValueError(
                f"DepthAnythingTower received pixel values in [-{peak:.3f}, {peak:.3f}]. "
                "That is SigLIP's 0.5 mean/std normalisation, not ImageNet. Set "
                "image_preprocess='depth_anything' on the data config AND on "
                "build_eval_transforms(); see gr00t/model/transforms_idm.py."
            )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Hydra entry point; mirrors `SiglipModel.from_pretrained`'s signature."""
        from transformers import DepthAnythingForDepthEstimation

        model = DepthAnythingForDepthEstimation.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        # Only the DINOv2 trunk is wanted; the DPT neck and depth head are discarded, which
        # is what "passed through the depth backbone" means.
        return cls(model.backbone)

    def forward(self, *args, **kwargs):  # pragma: no cover - action head calls vision_model
        raise NotImplementedError("call .vision_model(pixel_values)")
