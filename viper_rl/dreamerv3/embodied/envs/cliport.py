from collections import defaultdict
import functools
import os
import cv2

import embodied
import numpy as np
from embodied.envs import from_gym

# from . import from_gym


from cliport.cliport import tasks
from cliport.cliport.environments.environment import Environment
import cliport

CLIPORT_PATH = os.path.dirname(cliport.__file__)


class Cliport(embodied.Env):
    def __init__(self, task, size=(128, 128), allow_gripper_rot=False, **kwargs):
        self.env = Environment(
            f"{CLIPORT_PATH}/cliport/environments/assets/",
            disp=False,
            shared_memory=False,
            hz=480,
            record_cfg=None,
        )
        self.allow_gripper_rot = allow_gripper_rot
        self.task = tasks.names[task.replace("_", "-")]()
        self.task.mode = "train"
        self.env.set_task(self.task)
        self._done = True
        self._info = None

    @property
    def info(self):
        return self._info

    @functools.cached_property
    def obs_space(self):
        spaces = {}
        for i, space in enumerate(self.env.observation_space.spaces["color"]):
            spaces[f"image_{i}"] = embodied.Space(
                np.uint8, space.shape, space.low, space.high
            )
            spaces[f"image_small_{i}"] = embodied.Space(
                space.dtype,
                (64, 64, 3),
                np.zeros((64, 64, 3)).astype(np.uint8),
                255 * np.ones((64, 64, 3)).astype(np.uint8),
            )
        for i, space in enumerate(self.env.observation_space.spaces["depth"]):
            spaces[f"depth_{i}"] = embodied.Space(
                space.dtype,
                space.shape + (1,),
                space.low[..., None],
                space.high[..., None],
            )
        return {
            **spaces,
            "reward": embodied.Space(np.float32),
            "is_first": embodied.Space(bool),
            "is_last": embodied.Space(bool),
            "is_terminal": embodied.Space(bool),
        }

    @functools.cached_property
    def act_space(self):
        act_low = np.array([])
        act_high = np.array([])
        for action_tuple in self.env.action_space.spaces.values():
            act_low = np.append(act_low, action_tuple[0].low)
            act_high = np.append(act_high, action_tuple[0].high)
            if self.allow_gripper_rot:
                act_low = np.append(act_low, action_tuple[1].low)
                act_high = np.append(act_high, action_tuple[1].high)

        spaces = {}
        spaces["action"] = embodied.Space(np.float32, act_low.shape, act_low, act_high)
        spaces["reset"] = embodied.Space(bool)
        return spaces

    def _extract_ims_from_obs(self, obs):
        new_obs = {}
        for i, im in enumerate(obs["color"]):
            new_obs[f"image_{i}"] = im
            new_obs[f"image_small_{i}"] = cv2.resize(im, (64, 64))
        for i, d in enumerate(obs["depth"]):
            new_obs[f"depth_{i}"] = d[..., None]
        return new_obs

    def step(self, action):
        if action["reset"] or self._done:
            self._done = False
            self.env.set_task(self.task)
            obs = self.env.reset()
            info = self.env.info
            return self._obs(obs, 0.0, is_first=True, lang_goal=info["lang_goal"])

        idx = 0
        action_dict = {}
        for key, action_tuple in self.env.action_space.spaces.items():
            action_dict[key] = []
            for action_space in action_tuple:
                if action_space.shape[0] < 4 or self.allow_gripper_rot:
                    action_dict[key].append(
                        action["action"][idx : idx + action_space.shape[0]]
                    )
                    idx += action_space.shape[0]
                else:
                    action_dict[key].append(np.array([0.0, 0.0, 0.0, 1.0]))
            rot = action_dict[key][-1]
            action_dict[key][-1] = rot / np.linalg.norm(rot)

        info = self.env.info
        obs, reward, self._done, self._info = self.env.step(action_dict)
        return self._obs(
            obs,
            reward,
            is_last=bool(self._done),
            is_terminal=bool(self._info.get("is_terminal", self._done)),
            lang_goal=info["lang_goal"],
        )

    def _obs(
        self,
        obs,
        reward,
        is_first=False,
        is_last=False,
        is_terminal=False,
        lang_goal=None,
    ):
        obs = self._extract_ims_from_obs(obs)
        obs = {k: np.asarray(v) for k, v in obs.items()}
        if lang_goal is not None:
            obs["lang_goal"] = lang_goal
        obs.update(
            reward=np.float32(reward),
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal,
        )
        return obs

    def render(self):
        image = self.env.render("rgb_array")
        assert image is not None
        return image
