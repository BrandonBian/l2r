import logging
import subprocess
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

from gym import Wrapper
import gym

from tianshou.data import ReplayBuffer
from tianshou.data import Collector
from tianshou.policy import BasePolicy
from tianshou.utils.net.common import Net
from tianshou.env import DummyVectorEnv

from distrib_l2r.api import BufferMsg
from distrib_l2r.api import EvalResultsMsg
from distrib_l2r.api import InitMsg
from distrib_l2r.api import ParameterMsg
from distrib_l2r.utils import send_data
from src.constants import Task

from l2r import build_env

from src.config.yamlize import create_configurable, NameToSourcePath, yamlize
from src.constants import DEVICE
from src.utils.envwrapper import EnvContainer
import numpy as np


class AsnycWorker:
    """An asynchronous worker"""

    def __init__(
        self,
        learner_address: Tuple[str, int],
        buffer_size: int = 5000,
        env_wrapper: Optional[Wrapper] = None,
        **kwargs,
    ) -> None:

        self.learner_address = learner_address
        self.buffer_size = buffer_size
        self.mean_reward = 0.0

        self.env = build_env(controller_kwargs={"quiet": True},
                             env_kwargs={
            "multimodal": True,
            "eval_mode": True,
            "n_eval_laps": 5,
            "max_timesteps": 5000,
            "obs_delay": 0.1,
            "not_moving_timeout": 50000,
            "reward_pol": "custom",
            "provide_waypoints": False,
            "active_sensors": [
                "CameraFrontRGB"
            ],
            "vehicle_params": False,
        },
            action_cfg={
            "ip": "0.0.0.0",
            "port": 7077,
            "max_steer": 0.3,
            "min_steer": -0.3,
            "max_accel": 6.0,
            "min_accel": -1,
        },
            camera_cfg=[
                {
                    "name": "CameraFrontRGB",
                    "Addr": "tcp://0.0.0.0:8008",
                    "Width": 512,
                    "Height": 384,
                    "sim_addr": "tcp://0.0.0.0:8008",
                }
        ]
        )

        self.encoder = create_configurable(
            "config_files/async_sac/encoder.yaml", NameToSourcePath.encoder
        )
        self.encoder.to(DEVICE)

        self.env.action_space = gym.spaces.Box(
            np.array([-1, -1]), np.array([1.0, 1.0]))
        self.env = EnvContainer(self.encoder, self.env)

        self.runner = create_configurable(
            "config_files/async_sac/worker.yaml", NameToSourcePath.runner
        )

    def work(self) -> None:
        """Continously collect data"""

        print("INIT")
        response = send_data(
            data=InitMsg(), addr=self.learner_address, reply=True)

        policy_id, policy, task = response.data["policy_id"], response.data["policy"], response.data["task"]
        print(f"{task} | Param. Ver. = {policy_id}")

        while True:
            """ Process request, collect data """
            if task == Task.TRAIN:
                pass
            else:
                buffer, result = self.collect_data(
                    policy_weights=policy, task=task)

            """ Send response back to learner """
            if task == Task.COLLECT:
                """ Collect data, send back replay buffer (BufferMsg) """
                response = send_data(
                    data=BufferMsg(data=buffer),
                    addr=self.learner_address,
                    reply=True
                )
                print(
                    f"{task} | Param. Ver. = {policy_id} | Collected Buffer = {len(buffer)}")

            elif task == Task.EVAL:
                """ Evaluate parameters, send back reward (EvalResultsMsg) """
                response = send_data(
                    data=EvalResultsMsg(data=result),
                    addr=self.learner_address,
                    reply=True,
                )
                reward = result["reward"]
                print(f"{task} | Param. Ver. = {policy_id} | Reward = {reward}")

            else:
                """ Train parameters on the obtained replay buffers, send back updated parameters (ParameterMsg) """
                pass

            policy_id, policy, task = response.data["policy_id"], response.data["policy"], response.data["task"]

    def collect_data(
        self, policy_weights: dict, task: Task
    ) -> Tuple[ReplayBuffer, Any]:
        """ Collect 1 episode of data (replay buffer OR reward) in the environment """
        buffer, result = self.runner.run(self.env, policy_weights, task)
        return buffer, result
