      
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import random
import numpy as np
import torch
import math
import argparse
from collections import deque
from scipy.spatial.transform import Rotation as R
from sanic import Sanic, response
from loguru import logger
from PIL import Image

# Import GR00T related modules instead of QwenACTAFormerInference
from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.eval.robot import RobotInferenceClient, RobotInferenceServer
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from eval.sim_gr00t.adaptive_ensemble import AdaptiveEnsembler
import cv2 as cv

app = Sanic("InferenceServer")

import base64
import cv2
import numpy as np
import requests
import PIL

debug = False

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_model_path", type=str, default="nvidia/GR00T-N1.5-3B", 
                        help="Path to the saved model")
    parser.add_argument("--obs_type", type=str, default="obs_camera", 
                        help="Observation type")
    parser.add_argument("--policy_setup", type=str, default="genmanip",
                        help="Policy setup type")
    parser.add_argument("--use_bf16", action="store_true", 
                        help="Use bfloat16 precision")
    parser.add_argument("--action_ensemble", action="store_true", default=True,
                        help="Use action ensemble")
    parser.add_argument("--adaptive_ensemble_alpha", type=float, default=0.1,
                        help="Adaptive ensemble alpha")
    parser.add_argument("--cfg_scale", type=float, default=1.5,
                        help="CFG scale for inference (legacy parameter)")
    parser.add_argument("--action_horizon", type=int, default=16,
                        help="Action horizon for GR00T")
    # 添加裁剪相关参数
    parser.add_argument("--crop_obs_camera", action="store_true", default=False,
                        help="Enable cropping for obs_camera images")
    parser.add_argument("--crop_y_start", type=int, default=50,
                        help="Crop Y start coordinate")
    parser.add_argument("--crop_y_end", type=int, default=480,
                        help="Crop Y end coordinate")
    parser.add_argument("--crop_x_start", type=int, default=170,
                        help="Crop X start coordinate")
    parser.add_argument("--crop_x_end", type=int, default=635,
                        help="Crop X end coordinate")
    parser.add_argument("--action_type", type=str, default="delta_ee_pose",
                        help="Action type: delta_ee_pose, abs_ee_pose, delta_qpos, abs_qpos")
    return parser


def request_image(address="0.0.0.0", port=5021):
    url = f"http://{address}:{port}/image"
    headers = {"Content-Type": "application/json"}
    response = requests.get(url, headers=headers)
    response_json = response.json()
    if debug:
        print("Response Image.")
    return response_json


def get_camera_data(address="0.0.0.0", port="5021"):
    images = request_image(address=address, port=port)
    
    # 处理两个摄像头的数据
    colors_0 = base64.b64decode(images["data"]["colors"][0])
    colors_0 = cv2.imdecode(np.frombuffer(colors_0, np.uint8), cv2.IMREAD_COLOR)
    
    colors_1 = base64.b64decode(images["data"]["colors"][1])
    colors_1 = cv2.imdecode(np.frombuffer(colors_1, np.uint8), cv2.IMREAD_COLOR)
    print(f"colors_0 shape: {colors_0.shape}")
    print(f"colors_1 shape: {colors_1.shape}")
    depth_0 = np.frombuffer(
        base64.b64decode(images["data"]["depths"][0]), dtype=np.float64
    ).reshape(480, 640)
    
    depth_1 = np.frombuffer(
        base64.b64decode(images["data"]["depths"][1]), dtype=np.float64
    ).reshape(480, 640)
    
    return [colors_0, colors_1], [depth_0, depth_1]

def pose_to_6d(pose, degrees=False):
    pose6d = np.zeros(6)
    pose6d[:3] = pose[:3, 3]
    pose6d[3:6] =  R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=degrees)
    return pose6d


class GR00TInference:
    def __init__(
        self,
        saved_model_path: str = 'nvidia/GR00T-N1.5-3B',
        policy_setup: str = "genmanip",
        action_scale: float = 1.0,
        action_horizon: int = 16,
        action_ensemble: bool = True,
        action_ensemble_horizon: int = None,
        adaptive_ensemble_alpha: float = 0.1,
        # Image processing parameters
        crop_obs_camera: bool = False,
        crop_y_start: int = 50,
        crop_y_end: int = 480,
        crop_x_start: int = 170,
        crop_x_end: int = 635,
        # Action parameters
        action_type: str = "delta_ee_pose",
        # Legacy parameters for compatibility
        cfg_scale: float = 1.5,
        use_bf16: bool = False,
        **kwargs
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Set up policy setup defaults similar to GR00T_N1_5
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
        elif policy_setup == "genmanip":
            # For genmanip, use widowx_bridge as base config
            if action_ensemble_horizon is None:
                action_ensemble_horizon = 7
            self.sticky_gripper_num_repeat = 1
            data_config = DATA_CONFIG_MAP["oxe_bridge"]
            self.image_size = (224, 224)  # Keep original image size for genmanip
        else:
            raise NotImplementedError(
                f"Policy setup {policy_setup} not supported for GR00T models."
            )
        
        # Store core configuration
        self.policy_setup = policy_setup
        self.model_path = saved_model_path
        self.action_horizon = action_horizon
        self.action_scale = action_scale
        
        # Set up embodiment tag
        EMBODIMENT_TAG = "new_embodiment"
        
        # Get modality config and transform
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        print(f"*** policy_setup: {policy_setup} ***")
        
        # Load the GR00T policy
        self.policy = Gr00tPolicy(
            model_path=self.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=EMBODIMENT_TAG,
        )

        # Action ensemble configuration
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        
        # Action state tracking
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        # Task and history management
        self.task_description = None
        self.image_history = deque(maxlen=self.action_horizon)
        self.num_image_history = 0
        self.last_gripper_action = 0
        
        # Initialize action ensembler
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None

        # Image processing parameters
        self.crop_obs_camera = crop_obs_camera
        self.crop_y_start = crop_y_start
        self.crop_y_end = crop_y_end
        self.crop_x_start = crop_x_start
        self.crop_x_end = crop_x_end

        # Action configuration
        self.action_type = action_type

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.action_horizon)

    def reset(self, task_description: str = None) -> None:
        if task_description is not None:
            self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def _crop_image_if_needed(self, image: np.ndarray, is_obs_camera: bool = True) -> np.ndarray:
        """
        根据参数裁剪图像
        Args:
            image: 输入图像 (H, W, C)
            is_obs_camera: 是否为obs_camera类型的图像
        Returns:
            裁剪后的图像
        """
        if self.crop_obs_camera and is_obs_camera and image.shape[:2] == (480, 640):
            # 按照参数化的坐标进行裁剪
            cropped_image = image[self.crop_y_start:self.crop_y_end, 
                                self.crop_x_start:self.crop_x_end]
            return cropped_image
        return image

    def forward(self, obs_dict, language_instruction="", timestep=0):
        """
        Forward method to maintain compatibility with existing API
        """
        # Set task description if provided
        if language_instruction and language_instruction != self.task_description:
            self.task_description = language_instruction
            self.reset(language_instruction)

        # Extract images from obs_dict and apply cropping
        image_camera = obs_dict["obs_camera"]["color_image"]
        image_realsense = obs_dict["realsense"]["color_image"]
        current_ee_pose = obs_dict["robot"]["ee_pose_state"]
        
        # 确保图像是numpy数组
        if not isinstance(image_camera, np.ndarray):
            image_camera = np.array(image_camera, dtype=np.uint8)
        if not isinstance(image_realsense, np.ndarray):
            image_realsense = np.array(image_realsense, dtype=np.uint8)
        
        # 对obs_camera进行裁剪（如果启用）
        image_camera = self._crop_image_if_needed(image_camera, is_obs_camera=False)
        # realsense图像通常不需要裁剪，但如果需要可以设置is_obs_camera=True
        image_realsense = self._crop_image_if_needed(image_realsense, is_obs_camera=False)
        
        obs_dict = {
            # "video.base_view": self._resize_image(image_camera)[None],
            # "video.ego_view": self._resize_image(image_realsense)[None],
            "video.base_view": image_camera[None],
            "video.ego_view": image_realsense[None],
            "annotation.human.action.task_description": [language_instruction, ],
            "state.eef_position": np.array(current_ee_pose[:3])[None],
            "state.eef_rotation": np.array(current_ee_pose[3:6])[None],
        }

        # Get action from GR00T policy
        raw_actions = self.policy.get_action(obs_dict)
        new_raw_actions = []
        delta_eef_position = np.array(raw_actions["action.delta_eef_position"])
        delta_eef_rotation = np.array(raw_actions["action.delta_eef_rotation"])
        target_gripper = np.array(raw_actions["action.gripper_close"])
        new_raw_actions.append(delta_eef_position)
        new_raw_actions.append(delta_eef_rotation)
        new_raw_actions.append(target_gripper)
        new_raw_actions = np.array(new_raw_actions).transpose(1, 0)

        if self.action_ensemble:
            ensembled_actions = self.action_ensembler.ensemble_action(new_raw_actions)[None]
        ensembled_actions[:, 6] = np.where(ensembled_actions[:, 6] < 0.5, 0, 1) 
        ensembled_actions[:, 6] = ensembled_actions[:, 6] * 2 - 1
        action_to_use = ensembled_actions[0]


        target_eepose_delta = action_to_use[:6]
        target_gripper = action_to_use[6]
        
        is_terminal = -1.0
        return target_eepose_delta, target_gripper, is_terminal

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            image = np.array(image, dtype=np.uint8)
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image


def generate_action(data, agent, obs_type):
    obs = {}
    obs["robot"] = {"ee_pose_state": np.array(data["current_pose"])}
    images, depth = get_camera_data()
    print(f"images: {images[0].shape}")
    
    # 确保图像数据是numpy数组 hack
    image_realsense = images[0] if isinstance(images[0], np.ndarray) else np.array(images[0], dtype=np.uint8)
    image_camera = images[1] if isinstance(images[1], np.ndarray) else np.array(images[1], dtype=np.uint8)
    
    obs["obs_camera"] = {"color_image": image_camera}
    obs["realsense"] = {"color_image": image_realsense}
    
    goal = data["instruction"]
    timestep = data["timestep"]
    reset = data["reset"]
    PIL.Image.fromarray(image_camera).save(f"/home/pjlab/wangfangjing/llavavla/eval/deployment/debug/obs_camera{timestep:04d}.jpg")
    PIL.Image.fromarray(image_realsense).save(f"/home/pjlab/wangfangjing/llavavla/eval/deployment/debug/realsense_{timestep:04d}.jpg")
    if reset:
        agent.reset(goal)
    
    import time 
    time1 = time.time()
    output, gripper, _ = agent.forward(obs, goal, timestep)
    time2 = time.time()
    print(f"Time taken for inference: {time2 - time1:.4f} seconds")
    result = np.array(output)
    logger.debug(f"Output gripper: {gripper}")
    return result, gripper


def process_data(data, agent, obs_type, gripper_type):
    try:
        processed_eepose_delta, gripper = generate_action(data, agent, obs_type)
        if processed_eepose_delta is None:
            return {"message": "No action generated!"}
        if gripper_type == "panda_hand":
            eepose_action = processed_eepose_delta.tolist() + ([0.4, 0.4] if gripper == -1 else [0.0, 0.0])
        elif gripper_type == "robotiq":
            eepose_action = processed_eepose_delta.tolist() + ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0] if gripper == -1 else [math.pi, math.pi, 0.0, 0.0, 0.0, 0.0])
        return {"eepose_action": eepose_action}
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(str(e))
        return {"error": str(e)}

def formulate_input(data):
    state_pose = np.array(data["current_pose"])  # Shape: (N, 4, 4)
    state_xyz = state_pose[:3, 3]           # Extract translation (N, 3)
    state_rotation_matrices = state_pose[:3, :3]  # Extract rotation matrices (N, 3, 3)
    state_rot = R.from_matrix(state_rotation_matrices)
    state_rpy = state_rot.as_euler('xyz', degrees=False)  # Convert to RPY (N, 3)
    state_gripper_raw = np.array(data['current_gripper_width'])  # Gripper state (N, 1)
    state_gripper = (state_gripper_raw >= 0.04).astype(np.float32).reshape(-1, 1)  # Binary gripper state (N, 1)
    
    logger.debug(f"state_gripper: {state_gripper}")

    obs_abs_ee_pose = np.hstack((state_xyz, state_rpy, state_gripper[0]))  # Shape: (N, 7)
    return obs_abs_ee_pose


# Default arguments for GR00TInference - can be modified as needed
args = get_parser().parse_args()
agent = GR00TInference(
    saved_model_path=args.saved_model_path,
    policy_setup=args.policy_setup,
    use_bf16=args.use_bf16,
    action_ensemble=args.action_ensemble,
    adaptive_ensemble_alpha=args.adaptive_ensemble_alpha,
    cfg_scale=args.cfg_scale,
    action_horizon=args.action_horizon,
    # 添加裁剪参数
    crop_obs_camera=args.crop_obs_camera,
    crop_y_start=args.crop_y_start,
    crop_y_end=args.crop_y_end,
    crop_x_start=args.crop_x_start,
    crop_x_end=args.crop_x_end,
    # 添加 action_type
    action_type=args.action_type,
)
obs_type = args.obs_type


@app.post("/infer")
async def infer(request):
    data = request.json
    print(f"Received current pose(ee): {data['data']['current_pose']}")
    
    obs_abs_ee_pose = formulate_input(data['data'])
    data["current_pose"] = obs_abs_ee_pose
    actions = process_data(data, agent, obs_type, "robotiq")
    
    print(f"eepose actions: {actions}")
    print(f"action_type: {agent.action_type}")  # 添加这行来确认action_type
    
    logger.debug(f"eepose actions: {actions}")
    
    target_gripper = actions['eepose_action'][6]
    if target_gripper == 0.0:
        gripper_width = 0.08
    else:
        gripper_width = 0.0
    print(f"gripper_width: {gripper_width}")
    delta_eepose = actions['eepose_action'][:6]  # xyz + rpy (6维)
    return response.json({"target_eepose": (gripper_width, delta_eepose)})


if __name__ == "__main__":
    # 设置单进程模式，避免多进程启动冲突
    app.run(host="0.0.0.0", port=25553, workers=1, single_process=True)
