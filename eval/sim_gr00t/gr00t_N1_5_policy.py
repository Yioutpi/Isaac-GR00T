"""
cogact_policy.py

"""
from collections import deque
from typing import Optional, Sequence
import os
from PIL import Image
import torch
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from transforms3d.euler import euler2axangle
from scipy.spatial.transform import Rotation as R

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.eval.robot import RobotInferenceClient, RobotInferenceServer
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from .adaptive_ensemble import AdaptiveEnsembler


class GR00T_N1_5:
    def __init__(
        self,
        saved_model_path: str = 'nvidia/GR00T-N1.5-3B',
        action_scale: float = 1.0,
        policy_setup: str = None,
        action_horizon: int = 16,
        action_ensemble = True,
        action_ensemble_horizon: Optional[int] = None,
        adaptive_ensemble_alpha = 0.1,
    ) -> None:
        super().__init__()

        if policy_setup == "widowx_bridge":
            if action_ensemble_horizon is None:
                action_ensemble_horizon = 7
            self.sticky_gripper_num_repeat = 1
            data_config = DATA_CONFIG_MAP["oxe_bridge"]
            self.image_size = (256, 256)
        elif policy_setup == "google_robot":
            if action_ensemble_horizon is None:
                action_ensemble_horizon = 2
            self.sticky_gripper_num_repeat = 10
            data_config = DATA_CONFIG_MAP["oxe_rt1"]
            self.image_size = (256, 320)
        else:
            raise NotImplementedError(
                f"Policy setup {policy_setup} not supported for octo models. The other datasets can be found in the huggingface config.json file."
            )
        self.policy_setup = policy_setup
        self.model_path = saved_model_path
        self.action_horizon = action_horizon
        EMBODIMENT_TAG = "new_embodiment"

        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.policy = Gr00tPolicy(
            model_path=self.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=EMBODIMENT_TAG,
        )

        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None
        self.image_history = deque(maxlen=self.action_horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0
        self.last_gripper_action = 0
        self.action_scale = action_scale

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.action_horizon)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def step(
        self, image: np.ndarray, task_description: Optional[str] = None, tcp_pose: Optional[np.ndarray] = None, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
            tcp_pose: sapien.core.Pose(q, p)
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        # Convert sapien.core.Pose to position and quaternion
        if tcp_pose is not None:
            position = tcp_pose.p  # position as [x, y, z]
            quaternion = tcp_pose.q  # quaternion as [w, x, y, z]
        else:
            position = np.zeros(3)
            quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion

        if self.policy_setup == "widowx_bridge":
            # Convert quaternion to euler angles for widowx_bridge
            r = R.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])  # [x,y,z,w] format
            roll, pitch, yaw = r.as_euler('xyz')
            
            obs_dict = {
                "video.image_0": self._resize_image(image)[None],
                "annotation.human.action.task_description": [task_description, ],
                "state.x": np.array([position[0]])[None],  # Shape: (1, 1)
                "state.y": np.array([position[1]])[None], 
                "state.z": np.array([position[2]])[None],
                "state.roll": np.array([roll])[None],
                "state.pitch": np.array([pitch])[None],
                "state.yaw": np.array([yaw])[None],
                "state.pad": np.array([0.0])[None],
                "state.gripper": np.array([self.last_gripper_action], dtype=np.int64)[None],
            }
        elif self.policy_setup == "google_robot":
            obs_dict = {
                "video.image": self._resize_image(image)[None],
                "annotation.human.action.task_description": [task_description, ],
                "state.x": np.array([position[0]])[None],  # Shape: (1, 1)
                "state.y": np.array([position[1]])[None],
                "state.z": np.array([position[2]])[None],
                "state.rx": np.array([quaternion[1]])[None],  # x component of quaternion
                "state.ry": np.array([quaternion[2]])[None],  # y component of quaternion
                "state.rz": np.array([quaternion[3]])[None],  # z component of quaternion
                "state.rw": np.array([quaternion[0]])[None],  # w component of quaternion
                "state.gripper": np.array([self.last_gripper_action], dtype=np.int64)[None],
            }

        raw_actions = self.policy.get_action(obs_dict)
        
        # Stack xyz into world_vector, rpy into rotation_delta
        new_raw_actions = []
        for key in raw_actions.keys():
            if 'action' in key:
                new_raw_actions.append(np.array(raw_actions[key]))
        new_raw_actions = np.array(new_raw_actions).transpose(1, 0)

        if self.action_ensemble:
            ensembled_actions = self.action_ensembler.ensemble_action(new_raw_actions)
            action_to_use = ensembled_actions
        else:
            action_to_use = new_raw_actions[0]

        raw_action = {
            "world_vector": np.array(action_to_use[:3]),
            "rotation_delta": np.array(action_to_use[3:6]),
            "open_gripper": np.array(action_to_use[6:7]),
        }

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)

        roll, pitch, yaw = action_rotation_delta
        axes, angles = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = axes * angles
        action["rot_axangle"] = action_rotation_axangle * self.action_scale

        if self.policy_setup == "google_robot":
            action["gripper"] = 0
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
                self.previous_gripper_action = current_gripper_action
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            # fix a bug in the SIMPLER code here
            # self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action
                self.previous_gripper_action = current_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
        
        action["terminate_episode"] = np.array([0.0])
        if isinstance(action["gripper"], np.ndarray):
            self.last_gripper_action = action["gripper"][0]
        else:
            self.last_gripper_action = action["gripper"]
        return raw_action, action

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)