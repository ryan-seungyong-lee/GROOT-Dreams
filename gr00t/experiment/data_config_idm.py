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

from abc import ABC, abstractmethod

from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform, ModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.mirror import MirrorLeftRight
from gr00t.data.transform.state_action import (
    StateActionSinCosTransform,
    StateActionToTensor,
    StateActionTransform,
)
from gr00t.data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from gr00t.model.transforms_idm import GR00TIDMTransform


class BaseDataConfig(ABC):
    @abstractmethod
    def modality_config(self) -> dict[str, ModalityConfig]:
        pass

    @abstractmethod
    def transform(self) -> ModalityTransform:
        pass


###########################################################################################


class Gr1ArmsOnlyDataConfig(BaseDataConfig):
    video_keys = ["video.ego_view"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0, 16]
    action_indices = list(range(16))

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TIDMTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class So100DataConfig(BaseDataConfig):
    video_keys = ["video.webcam"]
    state_keys = ["state.main_shoulder_pan", "state.main_shoulder_lift", "state.main_elbow_flex", "state.main_wrist_flex", "state.main_wrist_roll", "state.main_gripper"]
    action_keys = ["action.main_shoulder_pan", "action.main_shoulder_lift", "action.main_elbow_flex", "action.main_wrist_flex", "action.main_wrist_roll", "action.main_gripper"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0, 16]
    action_indices = list(range(16))

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TIDMTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class Gr1FullUpperBodyDataConfig(BaseDataConfig):
    video_keys = ["video.front_view"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.waist",
        "state.neck",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.waist",
        "action.neck",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0, 16]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TIDMTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class BimanualPandaGripperDataConfig(BaseDataConfig):
    video_keys = [
        "video.right_wrist_view",
        "video.left_wrist_view",
        "video.front_view",
    ]
    state_keys = [
        "state.right_arm_eef_pos",
        "state.right_arm_eef_quat",
        "state.right_gripper_qpos",
        "state.left_arm_eef_pos",
        "state.left_arm_eef_quat",
        "state.left_gripper_qpos",
    ]
    action_keys = [
        "action.right_arm_eef_pos",
        "action.right_arm_eef_rot",
        "action.right_gripper_close",
        "action.left_arm_eef_pos",
        "action.left_arm_eef_rot",
        "action.left_gripper_close",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0, 16]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.right_arm_eef_pos": "min_max",
                    "state.right_gripper_qpos": "min_max",
                    "state.left_arm_eef_pos": "min_max",
                    "state.left_gripper_qpos": "min_max",
                },
                target_rotations={
                    "state.right_arm_eef_quat": "rotation_6d",
                    "state.left_arm_eef_quat": "rotation_6d",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.right_gripper_close": "binary",
                    "action.left_gripper_close": "binary",
                },
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TIDMTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class BimanualPandaHandDataConfig(BaseDataConfig):
    video_keys = [
        "video.right_wrist_view",
        "video.left_wrist_view",
        "video.ego_view",
    ]
    state_keys = [
        "state.right_arm_eef_pos",
        "state.right_arm_eef_quat",
        "state.right_hand",
        "state.left_arm_eef_pos",
        "state.left_arm_eef_quat",
        "state.left_hand",
    ]
    action_keys = [
        "action.right_arm_eef_pos",
        "action.right_arm_eef_rot",
        "action.right_hand",
        "action.left_arm_eef_pos",
        "action.left_arm_eef_rot",
        "action.left_hand",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0, 16]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.right_arm_eef_pos": "min_max",
                    "state.right_hand": "min_max",
                    "state.left_arm_eef_pos": "min_max",
                    "state.left_hand": "min_max",
                },
                target_rotations={
                    "state.right_arm_eef_quat": "rotation_6d",
                    "state.left_arm_eef_quat": "rotation_6d",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.right_hand": "min_max",
                    "action.left_hand": "min_max",
                },
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TIDMTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class SinglePandaGripperDataConfig(BaseDataConfig):
    video_keys = [
        "video.left_view",
        "video.right_view",
        "video.wrist_view",
    ]
    state_keys = [
        "state.end_effector_position_relative",
        "state.end_effector_rotation_relative",
        "state.gripper_qpos",
        "state.base_position",
        "state.base_rotation",
    ]
    action_keys = [
        "action.end_effector_position",
        "action.end_effector_rotation",
        "action.gripper_close",
        "action.base_motion",
        "action.control_mode",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0, 16]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.end_effector_position_relative": "min_max",
                    "state.end_effector_rotation_relative": "min_max",
                    "state.gripper_qpos": "min_max",
                    "state.base_position": "min_max",
                    "state.base_rotation": "min_max",
                },
                target_rotations={
                    "state.end_effector_rotation_relative": "rotation_6d",
                    "state.base_rotation": "rotation_6d",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.end_effector_position": "min_max",
                    "action.end_effector_rotation": "min_max",
                    "action.gripper_close": "binary",
                    "action.base_motion": "min_max",
                    "action.control_mode": "binary",
                },
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TIDMTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


###########################################################################################


class Gr1ArmsWaistDataConfig(Gr1ArmsOnlyDataConfig):
    video_keys = ["video.ego_view"]
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.waist",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.waist",
    ]
    language_keys = ["annotation.human.coarse_action"]
    observation_indices = [0, 16]
    action_indices = list(range(16))

    def modality_config(self):
        return super().modality_config()

    def transform(self):
        return super().transform()


###########################################################################################


class FrankaDataConfig(BaseDataConfig):
    video_keys = [
        "video.exterior_image_1_left_pad_res256_freq15",
        "video.exterior_image_2_left_pad_res256_freq15",
        "video.wrist_image_left_pad_res256_freq15",
    ]
    state_keys = [
        "state.eef_position",
        "state.eef_rotation",
        "state.gripper_position",
    ]
    action_keys = [
        "action.eef_position_delta",
        "action.eef_rotation_delta",
        "action.gripper_position",
    ]

    language_keys = ["annotation.language.language_instruction"]
    observation_indices = [0, 16]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.eef_position": "min_max",
                    "state.gripper_position": "min_max",
                },
                target_rotations={
                    "state.eef_rotation": "rotation_6d",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.eef_position_delta": "min_max",
                    "action.gripper_position": "binary",
                },
                target_rotations={
                    "action.eef_rotation_delta": "axis_angle",
                },
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TIDMTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)

###########################################################################################

class allex_thetwo_ck40_ego_ae40(BaseDataConfig):
    video_keys = ["video.camera_ego_left"]
    state_keys = [
        "state.right_arm_joints",
        "state.left_arm_joints",
        "state.right_hand_joints",
        "state.left_hand_joints",
        "state.neck_joints",
        "state.waist_joints",
    ]
    action_keys = [
        "action.right_arm_joints",
        "action.left_arm_joints",
        "action.right_hand_joints",
        "action.left_hand_joints",
        "action.neck_joints",
        "action.waist_joints",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0, 40] # NOTE : 반드시 2 frame을 Input으로 주어야함!
    action_indices = list(range(40))
    action_dim = 48

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        # NOTE: State는 IDM 모델에서 실제로 사용되지 않음 (Video만 사용)
        # GR1과 동일하게 state는 [0]만 로드 (일관성 유지)
        state_modality = ModalityConfig(
            delta_indices=[0],
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        language_modality = ModalityConfig(
            delta_indices=[0],
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms (NOTE: state는 IDM 모델에서 실제로 사용되지 않음)
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "q99" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "q99" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            # NOTE: state_horizon은 GR00TIDMTransform에서 실제로 사용되지 않음
            GR00TIDMTransform(
                state_horizon=1,  # state는 1개만 (실제로 모델에서 사용 안됨)
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=self.action_dim,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)
    

###########################################################################################

class allex_thetwo_ck40_ego_ae20(BaseDataConfig):
    video_keys = ["video.camera_ego_left"]
    state_keys = [
        "state.right_arm_joints",
        "state.left_arm_joints",
        "state.right_hand_joints",
        "state.left_hand_joints",
        "state.neck_joints",
        "state.waist_joints",
    ]
    action_keys = [
        "action.right_arm_joints",
        "action.left_arm_joints",
        "action.right_hand_joints",
        "action.left_hand_joints",
        "action.neck_joints",
        "action.waist_joints",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0, 20] # NOTE : 반드시 2 frame을 Input으로 주어야함!
    action_indices = list(range(20))
    action_dim = 48

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        # NOTE: Not use State in IDM
        state_modality = ModalityConfig(
            delta_indices=[0],
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        # NOTE: Not use Language in IDM
        language_modality = ModalityConfig(
            delta_indices=[0],
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms (NOTE: state는 IDM 모델에서 실제로 사용되지 않음)
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "q99" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "q99" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            # NOTE: state_horizon은 GR00TIDMTransform에서 실제로 사용되지 않음
            GR00TIDMTransform(
                state_horizon=1,  # state는 1개만 (실제로 모델에서 사용 안됨)
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=self.action_dim,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)
        
###########################################################################################

class allex_thetwo_ck40_ego_ae10(BaseDataConfig):
    video_keys = ["video.camera_ego_left"]
    state_keys = [
        "state.right_arm_joints",
        "state.left_arm_joints",
        "state.right_hand_joints",
        "state.left_hand_joints",
        "state.neck_joints",
        "state.waist_joints",
    ]
    action_keys = [
        "action.right_arm_joints",
        "action.left_arm_joints",
        "action.right_hand_joints",
        "action.left_hand_joints",
        "action.neck_joints",
        "action.waist_joints",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0, 10] # NOTE : 반드시 2 frame을 Input으로 주어야함!
    action_indices = list(range(10))
    action_dim = 48

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )

        # NOTE: Not use State in IDM
        state_modality = ModalityConfig(
            delta_indices=[0],
            modality_keys=self.state_keys,
        )

        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )

        # NOTE: Not use Language in IDM
        language_modality = ModalityConfig(
            delta_indices=[0],
            modality_keys=self.language_keys,
        )

        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

        return modality_configs

    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms (NOTE: state는 IDM 모델에서 실제로 사용되지 않음)
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "q99" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "q99" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            # NOTE: state_horizon은 GR00TIDMTransform에서 실제로 사용되지 않음
            GR00TIDMTransform(
                state_horizon=1,  # state는 1개만 (실제로 모델에서 사용 안됨)
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=self.action_dim,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)
    
class openarm_ego_ae20(BaseDataConfig):
    """OpenArm (Inspire RH56F1) IDM data config, action horizon = 20.

    Video-only IDM: input = 2 frames (t, t+20), output = 20-step action chunk.
    OpenArm has 28 DoF (no waist, unlike Allex). Keys are listed in natural joint-index
    order so the concatenated action vector equals the raw 28-d parquet action[0:28]:
        neck(0:2) | left_arm(2:9) | right_arm(9:16) | left_hand(16:22) | right_hand(22:28)
    Both ego cameras are fed as independent samples upstream (see prepare_openarm_idm.py),
    so the model still reads a single 'camera_ego_left' view here.
    """

    # Which vision-tower preprocessing to use. None = SigLIP processor (the baseline).
    image_preprocess = None

    video_keys = ["video.camera_ego_left"]
    state_keys = [
        "state.neck_joints",
        "state.left_arm_joints",
        "state.right_arm_joints",
        "state.left_hand_joints",
        "state.right_hand_joints",
    ]
    action_keys = [
        "action.neck_joints",
        "action.left_arm_joints",
        "action.right_arm_joints",
        "action.left_hand_joints",
        "action.right_hand_joints",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0, 20]  # NOTE: must be exactly 2 frames for IDM
    action_indices = list(range(20))
    action_dim = 28

    def modality_config(self) -> dict[str, ModalityConfig]:
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        # NOTE: state is NOT used by the IDM model; loaded only for the data pipeline.
        state_modality = ModalityConfig(
            delta_indices=[0],
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=[0],
            modality_keys=self.language_keys,
        )
        return {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }

    # Overridden by openarm_ego_ae20_mirror. Kept as a hook so the two configs differ by
    # exactly one transform and nothing else — this is an ablation.
    mirror_prob: float = 0.0
    # Camera pairs that trade places under the reflection. Empty for a single-camera
    # config; the stereo subclass sets it, without which mirroring would invert the
    # disparity sign — see MirrorLeftRight.video_swap_pairs.
    stereo_view_pairs: list = []

    def mirror_transforms(self) -> list[ModalityTransform]:
        if self.mirror_prob <= 0.0:
            return []
        return [
            MirrorLeftRight(
                p=self.mirror_prob,
                video_keys=self.video_keys,
                state_keys=self.state_keys,
                action_keys=self.action_keys,
                video_swap_pairs=self.stereo_view_pairs,
            )
        ]

    def transform(self) -> ModalityTransform:
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            # Mirroring goes here: the video is already a tensor, and the joint values
            # are still raw radians in their per-limb groups, which is what the
            # left-right map is defined on. It must precede StateActionTransform.
            *self.mirror_transforms(),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "q99" for key in self.state_keys},
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "q99" for key in self.action_keys},
            ),
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TIDMTransform(
                state_horizon=1,
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=self.action_dim,
                image_preprocess=self.image_preprocess,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class openarm_ego_ae20_mirror(openarm_ego_ae20):
    """openarm_ego_ae20 plus sagittal-plane mirroring at p=0.5.

    v4 was teleoperated almost entirely with the right arm — right-arm joint std runs
    2-7x the left arm's over the recordings — so the left arm gets very little
    supervision. Mirroring makes the two marginals identical, and on v3, which is
    already balanced, it acts as a plain x2 augmentation.

    Pair this with a dataset whose meta/stats.json was computed over the symmetrised
    distribution (prepare_openarm_idm_v3v4_10hz.py writes it as stats_mirror.json and
    stands it up as the -mirror dataset root). Using the unmirrored q99 here would clip
    every mirrored sample.
    """

    mirror_prob = 0.5


class openarm_ego_stereo_ae20(openarm_ego_ae20):
    """openarm_ego_ae20 reading BOTH ego cameras as two views of one sample.

    OpenArm has two synchronised head cameras. Until now the dataset builder emitted each
    source episode twice — once per camera — into the single camera_ego_left slot, so the
    model saw one eye per sample and never got stereo. Here the two cameras are two entries
    of video_keys, which is all the stack needs: ConcatTransform stacks them on a new view
    axis into [T, V, H, W, C], GR00TIDMTransform flattens (t v) to 4 images per sample
    ordered [t0L, t0R, t1L, t1R], and the view embedding indexes that joint (timestep,
    camera) position. Budget: 4 images x 16 tokens = 64 <= max_sequence_length 112, and
    max_num_views 6 >= 4, so IDM_dump/configs/openarm_ae20.yaml needs no change.

    Every video transform draws its randomness ONCE for all keys (VideoTransform.apply
    concatenates the keys, transforms, then splits), so the crop box and colour jitter are
    identical across both cameras and both timesteps. Stereo depends on that — per-eye
    crops would destroy the relative geometry the model is meant to exploit.

    Pair with a dataset built by prepare_openarm_idm_v3v4_10hz.py in stereo layout, where
    an episode carries both camera keys and the >=10Hz clips are INTERSECTED across the two
    views so the pair is simultaneous.
    """

    video_keys = ["video.camera_ego_left", "video.camera_ego_right"]


class openarm_ego_stereo_ae20_mirror(openarm_ego_stereo_ae20):
    """Stereo + sagittal mirroring at p=0.5.

    stereo_view_pairs is what makes the mirror correct here: reflecting the world moves the
    left eye to where the right eye was, so the two flipped frames must also exchange keys.
    Flipping each stream in place would hand the model a negated disparity — a stereo
    geometry that never occurs — for half of training.
    """

    mirror_prob = 0.5
    stereo_view_pairs = [("video.camera_ego_left", "video.camera_ego_right")]


class openarm_ego_stereo_ae20_depth_mirror(openarm_ego_stereo_ae20_mirror):
    """Stereo + mirror, preprocessed for the DepthAnything vision tower.

    The ONLY difference from openarm_ego_stereo_ae20_mirror is image_preprocess, which is
    the point: this is the B arm of a single-variable A/B against the SigLIP baseline
    idm_openarm_stereo_teleop20_mirror_bsz128_step30000_20260825 (test MAE 0.0358). Same
    dataset, same action_horizon 20, same mirroring, same crop/jitter -- only the backbone
    and the normalisation it needs.

    Pair with IDM_dump/configs/openarm_ae20_depth.yaml. Setting one without the other gives
    a DepthAnything tower fed SigLIP-normalised pixels (or the reverse); the tower's
    check_input_range() exists because that mismatch is otherwise silent.
    """

    image_preprocess = "depth_anything"


DATA_CONFIG_MAP = {
    "gr1_arms_waist": Gr1ArmsWaistDataConfig(),
    "gr1_arms_only": Gr1ArmsOnlyDataConfig(),
    "gr1_full_upper_body": Gr1FullUpperBodyDataConfig(),
    "bimanual_panda_gripper": BimanualPandaGripperDataConfig(),
    "bimanual_panda_hand": BimanualPandaHandDataConfig(),
    "single_panda_gripper": SinglePandaGripperDataConfig(),
    "so100": So100DataConfig(),
    "franka": FrankaDataConfig(),
    "allex_thetwo_ck40_ego_ae40": allex_thetwo_ck40_ego_ae40(),
    "allex_thetwo_ck40_ego_ae20": allex_thetwo_ck40_ego_ae20(),
    "allex_thetwo_ck40_ego_ae10": allex_thetwo_ck40_ego_ae10(),
    "openarm_ego_ae20": openarm_ego_ae20(),
    "openarm_ego_ae20_mirror": openarm_ego_ae20_mirror(),
    "openarm_ego_stereo_ae20": openarm_ego_stereo_ae20(),
    "openarm_ego_stereo_ae20_mirror": openarm_ego_stereo_ae20_mirror(),
    "openarm_ego_stereo_ae20_depth_mirror": openarm_ego_stereo_ae20_depth_mirror(),
}
