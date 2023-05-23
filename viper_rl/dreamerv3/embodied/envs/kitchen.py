from collections import defaultdict
import functools
import os

import embodied
import numpy as np
from . import from_gym


TASKS = [
    "microwave",
    "kettle",
    "slide cabinet",
    "hinge cabinet",
    "bottom burner",
    "light switch",
    "top burner",
]


class Kitchen(from_gym.FromGym):
    def __init__(self, task, size=(64, 64), **kwargs):
        assert task.startswith("kitchen"), f"wrong task name: {task}"

        if "MUJOCO_GL" not in os.environ:
            os.environ["MUJOCO_GL"] = "egl"
        if "MUJOCO_EGL_DEVICE_ID" not in os.environ:
            os.environ["MUJOCO_EGL_DEVICE_ID"] = os.environ.get(
                "CUDA_VISIBLE_DEVICES", "0"
            )

        import d4rl

        super().__init__(task, "state", "action", **kwargs)

        # Set frame buffer size to (64, 64) to reduce GPU memory usage.
        self._env.sim.model.vis.global_.offwidth = 64
        self._env.sim.model.vis.global_.offheight = 64

        self.solved_tasks = defaultdict(lambda: 0)
        self._size = size
        self._render = True

    @functools.cached_property
    def obs_space(self):
        spaces = super().obs_space.copy()
        if self._render:
            spaces["image"] = embodied.Space(np.uint8, self._size + (3,))
        for k in TASKS:
            spaces[f"log_complete_{k}"] = embodied.Space(bool)
        return spaces

    def step(self, action):
        if action["reset"] or self._done:
            self.solved_tasks = defaultdict(lambda: 0)
        ob = super().step(action)
        if self._render:
            from dm_control.mujoco import engine

            camera = engine.MovableCamera(self._env.sim, *self._size)
            camera.set_pose(
                distance=1.8, lookat=[-0.3, 0.5, 2.0], azimuth=90, elevation=-60
            )
            ob["image"] = camera.render()
        for task in TASKS:
            self.solved_tasks[task] = (
                self.solved_tasks[task] or task not in self._env.tasks_to_complete
            )
            ob[f"log_complete_{task}"] = self.solved_tasks[task]
        return ob

    def render(self):
        from dm_control.mujoco import engine

        camera = engine.MovableCamera(self._env.sim, *self._size)
        camera.set_pose(
            distance=1.8, lookat=[-0.3, 0.5, 2.0], azimuth=90, elevation=-60
        )
        return camera.render()
