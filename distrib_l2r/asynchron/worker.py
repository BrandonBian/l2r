import numpy as np
from src.utils.envwrapper import EnvContainer
from src.constants import DEVICE
from src.config.yamlize import create_configurable, NameToSourcePath, yamlize
import logging
import subprocess
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
import os

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
from distrib_l2r.utils import send_data
from src.constants import Task

logging.getLogger('').setLevel(logging.INFO)

# pip install git+https://github.com/learn-to-race/l2r.git@aicrowd-environment
# from l2r import build_env
# from l2r import RacingEnv

agent_name = os.getenv("AGENT_NAME")


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
        """
        self.env = build_env(controller_kwargs={"quiet": True},
           env_kwargs=
                   {
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
                       "vehicle_params":False,
                   },
           action_cfg=
                   {
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
            "config_files/async_sac_mountaincar/encoder.yaml", NameToSourcePath.encoder
        )
        self.encoder.to(DEVICE)

        self.env.action_space = gym.spaces.Box(
            np.array([-1, -1]), np.array([1.0, 1.0]))
        self.env = EnvContainer(self.encoder, self.env)
        """

        if agent_name == "mountain-car":
            self.env = gym.make("MountainCarContinuous-v0")
            self.runner = create_configurable(
                "config_files/async_sac_mountaincar/worker.yaml", NameToSourcePath.runner
            )
        elif agent_name == "bipedal-walker":
            self.env = gym.make("BipedalWalker-v3")
            self.runner = create_configurable(
                "config_files/async_sac_bipedalwalker/worker.yaml", NameToSourcePath.runner
            )
        else:
            print("Invalid Agent Name!")
            exit(1)

    def work(self) -> None:
        """Continously collect data"""
        logging.info(f">>> Worker Sending: [Init Message]")
        response = send_data(
            data=InitMsg(), addr=self.learner_address, reply=True)

        policy_id = response.data["policy_id"]
        policy = response.data["policy"]
        task = response.data["task"]
        logging.info(
            f"<<< Worker Receiving: [{task}] | Param. Ver. = {policy_id}")

        while True:
            if task != Task.TRAIN:
                buffer, result = self.process(
                    policy_weights=policy, task=task)

            if task == Task.COLLECT:
                """ Collect data, send back replay buffer (BufferMsg) """
                response = send_data(
                    data=BufferMsg(data=buffer),
                    addr=self.learner_address,
                    reply=True
                )

                logging.info(
                    f"Worker: [Task.COLLECT] | Buffer Size = {len(buffer)}")

            elif task == Task.EVAL:
                """ Evaluate parameters, send back reward (EvalResultsMsg) """
                response = send_data(
                    data=EvalResultsMsg(data=result),
                    addr=self.learner_address,
                    reply=True,
                )

                reward = result["reward"]

                logging.info(
                    f"Worker: [Task.EVAL] | Reward = {reward}")

            else:
                """ Train parameters on the obtained replay buffers, send back updated parameters (ParameterMsg) """
                logging.info(
                    f"Worker: [Task.TRAIN] | Not Implemented!")
                response = send_data(
                    data=InitMsg(), addr=self.learner_address, reply=True)

            policy_id, policy, task = response.data["policy_id"], response.data["policy"], response.data["task"]

    def process(
            self, policy_weights: dict, task: Task
    ) -> Tuple[ReplayBuffer, Any]:
        """Collect 1 episode of data in the environment"""

        if task == Task.TRAIN:
            pass
        else:
            buffer, result = self.runner.run(self.env, policy_weights, task)
            return buffer, result
