import copy
import embodied

import numpy as np
from rlbench.utils import name_to_task_class

try:
    from pyrep.errors import ConfigurationPathError, IKError
    from rlbench.backend.exceptions import InvalidActionError
except:
    pass


class RLBench(embodied.Env):
    def __init__(
        self,
        name,
        camera_keys,
        size=(64, 64),
        actions_min_max=None,
        shaped_rewards=False,
        robot="panda",
        restrict_to_box=False,
        start_gripper_low=False,
        use_magic_gripper=True,
        boundary_reward_penalty=False,
        use_rotation=True,
        terminate_on_success=False,
    ):
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import (
            EndEffectorPoseViaPlanning,
            EndEffectorPoseViaIK,
        )
        from rlbench.action_modes.gripper_action_modes import (
            Discrete,
        )
        from rlbench.environment import Environment
        from rlbench.observation_config import ObservationConfig
        from rlbench.tasks import (
            PhoneOnBase,
            PickAndLift,
            PickUpCup,
            PutRubbishInBin,
            TakeLidOffSaucepan,
            TakeUmbrellaOutOfUmbrellaStand,
            # MultiTaskMicrofridgesauce,
        )

        # we only support reach_target in this codebase
        obs_config = ObservationConfig()

        # Camera setups
        obs_config.front_camera.set_all(False)
        obs_config.wrist_camera.set_all(False)
        obs_config.left_shoulder_camera.set_all(False)
        obs_config.right_shoulder_camera.set_all(False)
        obs_config.overhead_camera.set_all(False)

        if "image_front" in camera_keys:
            obs_config.front_camera.rgb = True
            obs_config.front_camera.image_size = size

        if "image_wrist" in camera_keys:
            obs_config.wrist_camera.rgb = True
            obs_config.wrist_camera.image_size = size

        if "image_overhead" in camera_keys:
            obs_config.overhead_camera.rgb = True
            obs_config.overhead_camera.image_size = size

        obs_config.joint_forces = False
        obs_config.joint_positions = True
        obs_config.joint_velocities = True
        obs_config.task_low_dim_state = True
        obs_config.gripper_touch_forces = False
        obs_config.gripper_pose = True
        obs_config.gripper_open = True
        obs_config.gripper_matrix = False
        obs_config.gripper_joint_positions = True
        self._use_rotation = use_rotation

        env = Environment(
            action_mode=MoveArmThenGripper(
                arm_action_mode=EndEffectorPoseViaIK(use_rotation),
                gripper_action_mode=Discrete(attach_grasped_objects=use_magic_gripper),
            ),
            obs_config=obs_config,
            headless=True,
            shaped_rewards=shaped_rewards,
            robot_setup=robot,
            # static_positions=True,
        )
        env.launch()

        if name == "phone_on_base":
            task = PhoneOnBase
        elif name == "pick_and_lift":
            task = PickAndLift
        elif name == "pick_up_cup":
            task = PickUpCup
        elif name == "put_rubbish_in_bin":
            task = PutRubbishInBin
        elif name == "take_lid_off_saucepan":
            task = TakeLidOffSaucepan
        elif name == "take_umbrella_out_of_umbrella_stand":
            task = TakeUmbrellaOutOfUmbrellaStand
        elif name == "multi_task_microfridgesauce":
            task = MultiTaskMicrofridgesauce
        else:
            task = name_to_task_class(name)
        self._env = env
        self._task = env.get_task(task)
        self.task_name = name

        _, obs = self._task.reset()
        task_low_dim = obs.task_low_dim_state.shape[0]
        self._state_dim = obs.get_low_dim_data().shape[0] - 14 - task_low_dim
        self._prev_obs, self._prev_reward = None, None
        self._ep_success = None
        self._terminate_on_success = terminate_on_success

        self._size = size
        self._shaped_rewards = shaped_rewards
        self._camera_keys = camera_keys
        self._restrict_to_box = restrict_to_box
        self._start_gripper_low = start_gripper_low
        self._use_magic_gripper = use_magic_gripper
        self._boundary_reward_penalty = boundary_reward_penalty

        if actions_min_max:
            self.register_min_max(actions_min_max)
        else:
            self.low = np.array([-0.03, -0.03, -0.03])
            self.high = np.array([0.03, 0.03, 0.03])
            if self._use_rotation:
                self.low = np.array([-0.01833681, -0.02734856, -0.02796607])
                self.high = np.array([0.01568874, 0.0322583, 0.02788239])
                self.rot_low = np.array([-0.06, -0.03910604, -0.06])
                self.rot_high = np.array([0.06, 0.04957092, 0.06])

    @property
    def obs_space(self):
        spaces = {
            "reward": embodied.Space(np.float32),
            "is_first": embodied.Space(bool),
            "is_last": embodied.Space(bool),
            "is_terminal": embodied.Space(bool),
        }
        for key in self._camera_keys:
            spaces[key] = embodied.Space(np.uint8, (*self._size, 3))
        return spaces

    def register_min_max(self, actions_min_max):
        self.low, self.high = actions_min_max
        print(f"action space lower bound set to: {self.low}")
        print(f"action space upper bound set to: {self.high}")

    @property
    def act_space(self):
        # First 3 actions are the delta change in position. Last action is the gripper.
        assert self.low is not None
        if self.low.shape[0] == 3:
            self.low = np.hstack([self.low, [0.0]])
            self.high = np.hstack([self.high, [1.0]])
        low = np.array([-1.0, -1.0, -1.0, -1.0])
        high = np.array([1.0, 1.0, 1.0, 1.0])
        if self._use_rotation and self.low.shape[0] < 7:
            self.low = np.hstack([self.low, self.rot_low])
            self.high = np.hstack([self.high, self.rot_high])
        if self._use_rotation:
            low = np.hstack([low, [-1.0, -1.0, -1.0]])
            high = np.hstack([high, [1.0, 1.0, 1.0]])
        action = embodied.Space(np.float32, (low.shape[0],), low, high)
        return {"action": action, "reset": embodied.Space(bool)}

    def unnormalize(self, a):
        # Un-normalize gripper pose normalized to [-1, 1]
        assert self.low is not None
        pose = a[:3]
        pose = (pose + 1) / 2 * (self.high[:3] - self.low[:3]) + self.low[:3]

        if self._restrict_to_box:
            # Handle box overflow
            init_z = self._init_pose[2]
            curr_pose = self._task._task.robot.arm.get_tip().get_pose()[:3]
            curr_x, curr_y, curr_z = curr_pose[0], curr_pose[1], curr_pose[2]
            delta_x, delta_y, delta_z = pose[0], pose[1], pose[2]
            if curr_x + delta_x >= self.x_max or curr_x + delta_x <= self.x_min:
                pose[0] = 0.0
            if curr_y + delta_y >= self.y_max or curr_y + delta_y <= self.y_min:
                pose[1] = 0.0
            if curr_z + delta_z >= init_z:
                pose[2] = 0.0
        else:
            # Manual handling of overflow in z axis
            curr_pose = self._task._task.robot.arm.get_tip().get_pose()[:3]
            curr_z = curr_pose[2]
            init_z = self._init_pose[2]
            delta_z = pose[2]

            if curr_z + delta_z >= init_z:
                pose[2] = 0.0

        # Un-normalize gripper action normalized to [-1, 1]
        gripper = a[3:4]
        gripper = (gripper + 1) / 2 * (self.high[3:4] - self.low[3:4]) + self.low[3:4]

        if self._use_rotation:
            target_pose = curr_pose + pose
            curr_quat = self._task._task.robot.arm.get_tip().get_pose()[3:]
            d_theta = (a[4:7] + 1) / 2 * (self.high[4:7] - self.low[4:7]) + self.low[
                4:7
            ]
            curr_theta = self.quat_to_theta(curr_quat)
            theta = curr_theta + d_theta
            quat = self.theta_to_quat(theta)
            quat = quat / np.linalg.norm(quat)

        else:
            target_pose = pose
            # Identity quaternion
            quat = np.array([0.0, 0.0, 0.0, 1.0])

        action = np.hstack([target_pose, quat, gripper])
        assert action.shape[0] == 8
        return action

    def step(self, action):
        if action["reset"]:
            return self.reset()

        assert np.isfinite(action["action"]).all(), action["action"]
        try:
            original_action = self.unnormalize(action["action"])
            _obs, _reward, _ = self._task.step(original_action)
            terminal = False
            success, _ = self._task._task.success()
            if success:
                self._ep_success = True
            self._prev_obs, self._prev_reward = _obs, _reward
            if not self._shaped_rewards:
                reward = float(self._ep_success)
            else:
                reward = _reward
        except ConfigurationPathError:
            _obs = self._prev_obs
            terminal = False
            success = False
            if not self._shaped_rewards:
                reward = float(self._ep_success)
            else:
                reward = self._prev_reward
        except (IKError, InvalidActionError):
            _obs = self._prev_obs
            success = False
            if self._boundary_reward_penalty:
                terminal = True
                reward = -0.05
            else:
                terminal = False
                if not self._shaped_rewards:
                    reward = float(self._ep_success)
                else:
                    reward = self._prev_reward

        _obs.joint_velocities = None
        _obs.joint_positions = None
        _obs.task_low_dim_state = None

        if self._terminate_on_success:
            terminal = terminal or success

        obs = {
            "reward": reward,
            "is_first": False,
            "is_last": terminal,
            "is_terminal": terminal,
        }
        for key in self._camera_keys:
            if key == "image_front":
                obs[key] = _obs.front_rgb
            if key == "image_wrist":
                obs[key] = _obs.wrist_rgb
            if key == "image_overhead":
                obs[key] = _obs.overhead_rgb
        self._time_step += 1

        return obs

    def reset(self):
        print(f"Reset in env {self.task_name}.")
        _, _obs = self._task.reset()

        self._prev_obs = _obs
        self._time_step = 0
        self._ep_success = False

        if self._restrict_to_box:
            if hasattr(self._task._task, "boundary"):
                local_bounds = self._task._task.boundary._boundaries[
                    0
                ]._boundary.get_bounding_box()[:4]
                spawn_center = self._task._task.boundary._boundaries[
                    0
                ]._boundary.get_position()[:2]
                self.x_min = spawn_center[0] + local_bounds[0]
                self.x_max = spawn_center[0] + local_bounds[1]
                self.y_min = spawn_center[1] + local_bounds[2]
                self.y_max = spawn_center[1] + local_bounds[3]
            else:
                self.x_min, self.x_max = 0.0988, 0.4988
                # self.y_min, self.y_max = -0.3333, 0.2666
                self.y_min, self.y_max = -0.28, 0.24
            print("Set workspace boundaries:")
            print(f"X min: {self.x_min} / max: {self.x_max}")
            print(f"Y min: {self.y_min} / max: {self.y_max}")

        self._init_pose = copy.deepcopy(
            self._task._task.robot.arm.get_tip().get_pose()[:3]
        )

        _obs.joint_velocities = None
        _obs.joint_positions = None
        _obs.task_low_dim_state = None

        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
        }
        for key in self._camera_keys:
            if key == "image_front":
                obs[key] = _obs.front_rgb
            if key == "image_wrist":
                obs[key] = _obs.wrist_rgb
            if key == "image_overhead":
                obs[key] = _obs.overhead_rgb

        if self._start_gripper_low:
            action = np.zeros(self.act_space["action"].shape)
            action[2] = -1.0
            for _ in range(4):
                obs = self.step({"action": action, "reset": False})
                obs["reward"] = 0.0
                obs["is_first"] = True
                obs["is_last"] = False
                obs["is_terminal"] = False
        self._init_pose = copy.deepcopy(
            self._task._task.robot.arm.get_tip().get_pose()[:3]
        )

        return obs

    def theta_to_quat(self, thetas):
        theta1, theta2, theta3 = thetas
        x1 = np.cos(theta1)
        x2 = np.sin(theta1) * np.cos(theta2)
        x3 = np.sin(theta1) * np.sin(theta2) * np.cos(theta3)
        x4 = np.sin(theta1) * np.sin(theta2) * np.sin(theta3)
        quat = np.hstack([x2, x3, x4, x1])
        return quat

    def quat_to_theta(self, quat):
        x2, x3, x4, x1 = quat
        theta1 = np.arccos(x1 / np.sqrt(x4**2 + x3**2 + x2**2 + x1**2))
        theta2 = np.arccos(x2 / np.sqrt(x4**2 + x3**2 + x2**2))
        theta3 = np.arccos(x3 / np.sqrt(x4**2 + x3**2))
        if x4 < 0:
            theta3 = 2 * np.pi - theta3
        thetas = np.hstack([theta1, theta2, theta3])
        return thetas
