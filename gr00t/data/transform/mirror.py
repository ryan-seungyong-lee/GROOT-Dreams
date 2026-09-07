# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sagittal-plane mirroring for bimanual humanoid data.

An IDM learns p(a_{t:t+H} | I_t, I_{t+H}). Reflecting the world about the robot's
sagittal plane maps a valid (observation, action) pair to another valid one, provided
the robot is bilaterally symmetric — so flipping the frames and mirroring the action is
a label-equivariant augmentation, the bimanual analogue of a horizontal flip in image
classification.

It is worth more than a plain doubling of data when one arm dominates the recordings:
mirroring makes the left and right marginals identical, so the under-used arm inherits
the busy arm's supervision.

Two things this transform depends on, both of which are the caller's responsibility:

  * The video and the action must be flipped from the *same* coin toss. That is why this
    is one transform over the whole sample dict rather than a video transform plus a
    state/action transform — VideoHorizontalFlip would draw its own.
  * Normalisation statistics must come from the symmetrised distribution. Under p=0.5
    the mirrored left-arm values follow the original right arm's spread, and normalising
    those with the unmirrored left arm's q99 silently clips them.
"""

from typing import Any, ClassVar

import numpy as np
import torch
from pydantic import Field, model_validator

from gr00t.data.transform.base import ModalityTransform


class MirrorLeftRight(ModalityTransform):
    """Mirror video and state/action left-right with probability ``p``.

    Operates on the per-group modality keys (``state.left_arm_joints`` and friends)
    before ConcatTransform, and on raw joint values before normalisation.

    The default suffix maps describe OpenArm (Inspire RH56F1): neck 2 + arms 7+7 +
    hands 6+6. ``arm_sign`` was determined from the recordings rather than a URDF — over
    100 v3 and 100 v4 episodes the frame-0 home pose satisfies left[j] == -right[j] for
    arm joints 1,2,3,5,6,7 and left[4] == +right[4]. Joint 4 is the elbow, whose axis is
    perpendicular to the sagittal plane and therefore survives the reflection unchanged.
    Hand joints are non-negative flexion magnitudes on mirror-image hardware, so they
    swap without a sign change; neck yaw negates while neck pitch does not.
    """

    p: float = Field(default=0.5, description="Probability of mirroring a sample.")
    video_keys: list[str] = Field(default_factory=list)
    state_keys: list[str] = Field(default_factory=list)
    action_keys: list[str] = Field(default_factory=list)
    video_swap_pairs: list[tuple[str, str]] = Field(
        default_factory=list,
        description="Camera pairs that exchange places under the reflection, e.g. "
                    "[('video.camera_ego_left', 'video.camera_ego_right')]. Required for "
                    "stereo; leave empty for a single-camera config.",
    )

    neck_suffix: str = Field(default="neck_joints")
    neck_sign: list[float] = Field(default=[1.0, -1.0], description="pitch, yaw")
    arm_suffixes: tuple[str, str] = Field(default=("left_arm_joints", "right_arm_joints"))
    arm_sign: list[float] = Field(default=[-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0])
    hand_suffixes: tuple[str, str] = Field(default=("left_hand_joints", "right_hand_joints"))
    hand_sign: list[float] | None = Field(default=None, description="None == no sign change")

    _MODALITIES: ClassVar[tuple[str, ...]] = ("state", "action")

    apply_to: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _fill_apply_to(self):
        if not self.apply_to:
            self.apply_to = [*self.video_keys, *self.state_keys, *self.action_keys]
        seen: set[str] = set()
        for a, b in self.video_swap_pairs:
            assert a != b, f"video_swap_pairs entry swaps a key with itself: {a}"
            for k in (a, b):
                assert k in self.video_keys, f"{k} in video_swap_pairs but not in video_keys"
                assert k not in seen, f"{k} appears in more than one video_swap_pairs entry"
                seen.add(k)
        return self

    # -- primitives ---------------------------------------------------------

    @staticmethod
    def _flip_video(x):
        """Horizontal flip for either layout the pipeline may hand us.

        After VideoToTensor a clip is a torch tensor [T, C, H, W] (width last); before
        it, a numpy array [T, H, W, C] (width second to last).
        """
        if isinstance(x, torch.Tensor):
            return torch.flip(x, dims=[-1])
        if isinstance(x, np.ndarray):
            return np.ascontiguousarray(x[..., ::-1, :])
        raise TypeError(f"Unsupported video type for mirroring: {type(x)}")

    @staticmethod
    def _scale(x, sign):
        if sign is None:
            return x
        if isinstance(x, torch.Tensor):
            return x * torch.as_tensor(sign, dtype=x.dtype, device=x.device)
        return x * np.asarray(sign, dtype=x.dtype)

    def _swap_pair(self, data: dict[str, Any], modality: str,
                   suffixes: tuple[str, str], sign) -> None:
        left, right = (f"{modality}.{s}" for s in suffixes)
        if left not in data and right not in data:
            return
        assert left in data and right in data, (
            f"MirrorLeftRight needs both halves of the pair to swap them; "
            f"got {left in data=} {right in data=}. Available: {sorted(data)}"
        )
        a, b = data[left], data[right]
        assert a.shape == b.shape, f"{left} {a.shape} and {right} {b.shape} must match"
        data[left], data[right] = self._scale(b, sign), self._scale(a, sign)

    # -- transform ----------------------------------------------------------

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        # Eval must stay deterministic, otherwise a held-out metric moves for reasons
        # that have nothing to do with the model.
        if not self.training or self.p <= 0.0:
            return data
        # One draw for the whole sample: the frames and the action have to agree.
        if float(torch.rand(())) >= self.p:
            return data

        for key in self.video_keys:
            if key in data:
                data[key] = self._flip_video(data[key])

        # A sagittal reflection also swaps which physical camera is which eye: a camera at
        # x = -d in the reflected world sees what the camera at x = +d saw, mirrored. So
        # I'_left = flip(I_right) and I'_right = flip(I_left). Flipping each stream in place
        # (the loop above, alone) negates the disparity sign instead, which is a stereo
        # geometry that never occurs in real data — at p=0.5 that would feed half the
        # training set impossible pairs. Empty by default, so the mono path is unchanged.
        for a, b in self.video_swap_pairs:
            if a in data and b in data:
                data[a], data[b] = data[b], data[a]

        for modality in self._MODALITIES:
            neck = f"{modality}.{self.neck_suffix}"
            if neck in data:
                data[neck] = self._scale(data[neck], self.neck_sign)
            self._swap_pair(data, modality, tuple(self.arm_suffixes), self.arm_sign)
            self._swap_pair(data, modality, tuple(self.hand_suffixes), self.hand_sign)

        return data
