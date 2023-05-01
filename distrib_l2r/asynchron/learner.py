import logging
import queue
import random
import socketserver
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from tqdm import tqdm
import socket
import threading
from copy import deepcopy
import time
import sys
import os
from src.agents.base import BaseAgent
from src.config.yamlize import create_configurable, NameToSourcePath, yamlize
from src.loggers.WanDBLogger import WanDBLogger
from src.constants import Task

from distrib_l2r.api import BufferMsg
from distrib_l2r.api import InitMsg
from distrib_l2r.api import EvalResultsMsg
from distrib_l2r.api import PolicyMsg
from distrib_l2r.api import ParameterMsg
from distrib_l2r.utils import receive_data
from distrib_l2r.utils import send_data

logging.getLogger('').setLevel(logging.INFO)

SEND_BATCH = 300


class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    """Request handler thread created for every request"""

    def handle(self) -> None:
        """ReplayBuffers are not thread safe - pass data via thread-safe queues"""
        msg = receive_data(self.request)

        # Received a replay buffer from a worker
        # Add this to buff
        if isinstance(msg, BufferMsg):
            print(f"[COLLECT]   | Buffer Size = {len(msg.data)}")
            self.server.buffer_queue.put(msg.data)

        # Received an init message from a worker
        # Immediately reply with the most up-to-date policy
        elif isinstance(msg, InitMsg):
            print("INIT")

        # Received evaluation results from a worker
        elif isinstance(msg, EvalResultsMsg):
            print(f"[EVAL]      | Message = {msg.data}")
            self.server.wandb_logger.log(
                {
                    "reward": msg.data["reward"],
                    "Distance": msg.data["total_distance"],
                    "Time": msg.data["total_time"],
                    "Num infractions": msg.data["num_infractions"],
                    "Average Speed KPH": msg.data["average_speed_kph"],
                    "Average Displacement Error": msg.data["average_displacement_error"],
                    "Trajectory Efficiency": msg.data["trajectory_efficiency"],
                    "Trajectory Admissability": msg.data["trajectory_admissibility"],
                    "Movement Smoothness": msg.data["movement_smoothness"],
                    "Timestep per Sec": msg.data["timestep/sec"],
                    "Laps Completed": msg.data["laps_completed"],
                }
            )

        # Received trained parameters from a worker
        # Update current parameter with damping factors
        elif isinstance(msg, ParameterMsg):
            new_parameters = msg.data["parameters"]
            current_parameters = {k: v.cpu()
                                  for k, v in self.server.agent.state_dict().items()}

            assert set(current_parameters.keys()) == set(
                new_parameters.keys()), "Parameters from worker not matching learner's!"

            # Loop through the keys of the dictionaries and update the values of old_dict using the damping formula
            alpha = 0.8
            for key in current_parameters:
                old_value = current_parameters[key]
                new_value = new_parameters[key]
                updated_value = alpha * old_value + (1 - alpha) * new_value
                current_parameters[key] = updated_value

            self.server.agent.load_model(current_parameters)
            self.server.update_agent_queue()

            print(
                f"[TRAIN]     | Param. Mean = {sum((x.cpu().numpy()).mean() for x in current_parameters.values())}, Param. Std = {sum((x.cpu().numpy()).std() for x in current_parameters.values())}")

        # unexpected
        else:
            logging.warning(f"Received unexpected data: {type(msg)}")
            return

        # Reply to the request with an up-to-date policy
        start = time.time()
        msg = self.server.get_agent_dict()
        send_data(data=PolicyMsg(data=msg),
                  sock=self.request)
        if msg["task"] == Task.TRAIN:
            print(f"Timing      | Data sending time: {round(time.time() - start, 4)} s")


class AsyncLearningNode(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """A multi-threaded, offline, off-policy reinforcement learning server

    Args:
        policy: an intial Tianshou policy
        update_steps: the number of gradient updates for each buffer received
        batch_size: the batch size for gradient updates
        epochs: the number of buffers to receive before concluding learning
        server_address: the address the server runs on
        eval_freq: the likelihood of responding to a worker to eval instead of train
        save_func: a function for saving which is called while learning with
          parameters `epoch` and `policy`
        save_freq: the frequency, in epochs, to save
    """

    def __init__(
        self,
        agent: BaseAgent,
        update_steps: int = 100,
        batch_size: int = 128,  # Originally 128
        epochs: int = 500,  # Originally 500
        buffer_size: int = 1_000_000,  # Originally 1M
        server_address: Tuple[str, int] = ("0.0.0.0", 4444),
        save_func: Optional[Callable] = None,
        save_freq: Optional[int] = None,
        api_key: str = "",
    ) -> None:

        super().__init__(server_address, ThreadedTCPRequestHandler)
        self.update_steps = update_steps
        self.batch_size = batch_size
        self.epochs = epochs

        # Create a replay buffer
        self.buffer_size = buffer_size
        self.replay_buffer = create_configurable(
            "config_files/async_sac/buffer.yaml", NameToSourcePath.buffer
        )

        # Inital policy to use
        self.agent = agent
        self.agent_id = 1

        # The bytes of the policy to reply to requests with

        self.agent_params = {k: v.cpu()
                             for k, v in self.agent.state_dict().items()}

        # A thread-safe policy queue to avoid blocking while learning. This marginally
        # increases off-policy error in order to improve throughput.
        self.agent_queue = queue.Queue(maxsize=1)

        # A queue of buffers that have been received but not yet added to the learner's
        # main replay buffer
        self.buffer_queue = queue.LifoQueue(30)

        self.wandb_logger = WanDBLogger(
            api_key=api_key, project_name="test-project")
        # Save function, called optionally
        self.save_func = save_func
        self.save_freq = save_freq

    def get_agent_dict(self) -> Dict[str, Any]:
        """Get the most up-to-date version of the policy without blocking"""
        if not self.agent_queue.empty():
            try:
                self.agent_params = self.agent_queue.get_nowait()
            except queue.Empty:
                # non-blocking
                pass

        task = self.select_task()

        if task == Task.TRAIN:
            buffers_to_send = []

            start = time.time()
            for _ in range(SEND_BATCH):
                batch = self.replay_buffer.sample_batch()
                buffers_to_send.append(batch)
            
            print(f"Timing      | Data preparation time: {round(time.time() - start, 4)} s")

            msg = {
                "policy_id": self.agent_id,
                "policy": self.agent_params,
                "replay_buffer": buffers_to_send,
                "task": task
            }
        else:
            msg = {
                "policy_id": self.agent_id,
                "policy": self.agent_params,
                "task": task
            }

        return msg

    def update_agent_queue(self) -> None:
        """Update policy that will be sent to workers without blocking"""
        if not self.agent_queue.empty():
            try:
                # empty queue for safe put()
                _ = self.agent_queue.get_nowait()
            except queue.Empty:
                pass

        self.agent_queue.put({k: v.cpu()
                             for k, v in self.agent.state_dict().items()})
        self.agent_id += 1

    def learn(self) -> None:
        """The thread where thread-safe gradient updates occur"""
        epoch = 0
        while True:
            semibuffer = self.buffer_queue.get()
            print(
                f"Sampling    | Epoch = {epoch} -> Sampled Buffer = {len(semibuffer)} from Replay Buffer = {len(self.replay_buffer)}, where Buffer Queue = {self.buffer_queue.qsize()}")
            # Add new data to the primary replay buffer
            self.replay_buffer.store(semibuffer)
            epoch += 1

    def server_bind(self):
        # From https://stackoverflow.com/questions/6380057/python-binding-socket-address-already-in-use/18858817#18858817.
        # Tries to ensure reuse. Might be wrong.
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.server_address)

    def select_task(self):
        if len(self.replay_buffer) < 2048:
            # If replay buffer is empty, we need to collect more data
            return Task.COLLECT
        else:
            weights = [0.5, 0.1, 0.4]
            return random.choices([Task.TRAIN, Task.EVAL, Task.COLLECT], weights=weights)[0]
