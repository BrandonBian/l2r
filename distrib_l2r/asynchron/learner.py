import logging
import queue
import random
import socketserver
import threading
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from copy import deepcopy
import time
from tqdm import tqdm
import sys
import socket
import os
from src.agents.base import BaseAgent
from src.config.yamlize import create_configurable, NameToSourcePath, yamlize
from src.loggers.WanDBLogger import WanDBLogger

from distrib_l2r.api import BufferMsg
from distrib_l2r.api import InitMsg
from distrib_l2r.api import EvalResultsMsg
from distrib_l2r.api import PolicyMsg
from distrib_l2r.api import ParameterMsg
from distrib_l2r.utils import receive_data
from distrib_l2r.utils import send_data
from src.constants import Task

logging.getLogger('').setLevel(logging.INFO)
agent_name = os.getenv("AGENT_NAME")

# https://stackoverflow.com/questions/41653281/sockets-with-threadpool-server-python

# New message type: parameters + batch of data (worker learn on this data, batches from replay buffer), worker provides gradients back to the server
# The server applies all the changes - iterate through all the received gradients (quality or quantity matters)

# Sequential: collect + train, sending parameters instead of gradients, server average the parameters

TIMING = False
SEND_BATCH = 30


class ThreadPoolMixIn(socketserver.ThreadingMixIn):
    '''
    use a thread pool instead of a new thread on every request
    '''
    # numThreads = 50
    allow_reuse_address = True  # seems to fix socket.error on server restart

    def serve_forever(self):
        '''
        Handle one request at a time until doomsday.
        '''
        print('[X] Server is Running with No of Threads :- {}'.format(self.numThreads))
        # set up the threadpool
        self.requests = queue.Queue(self.numThreads)

        for x in range(self.numThreads):
            t = threading.Thread(target=self.process_request_thread)
            t.setDaemon(1)
            t.start()

        # server main loop
        while True:
            self.handle_request()
        self.server_close()

    def process_request_thread(self):
        '''
        obtain request from queue instead of directly from server socket
        '''
        while True:
            socketserver.ThreadingMixIn.process_request_thread(
                self, *self.requests.get())

    def handle_request(self):
        '''
        simply collect requests and put them on the queue for the workers.
        '''
        try:
            request, client_address = self.get_request()
        except socket.error:
            return
        if self.verify_request(request, client_address):
            self.requests.put((request, client_address))


class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    """Request handler thread created for every request"""

    def handle(self) -> None:
        """ReplayBuffers are not thread safe - pass data via thread-safe queues"""
        msg = receive_data(self.request)

        # Received a replay buffer from a worker
        # Add this to buff
        if isinstance(msg, BufferMsg):
            logging.info(
                f"<<< Learner Receiving: [Replay Buffer] | Buffer Size = {len(msg.data)}")
            self.server.buffer_queue.put(msg.data)

        # Received an init message from a worker
        # Immediately reply with the most up-to-date policy
        elif isinstance(msg, InitMsg):
            logging.info(f"<<< Learner Receiving: [Init Message]")

        # Received evaluation results from a worker
        # Log to Weights and Biases
        elif isinstance(msg, EvalResultsMsg):
            reward = msg.data["reward"]
            logging.info(
                f"<<< Learner Receiving: [Reward] | Reward = {reward}")
            self.server.wandb_logger.log_metric(
                reward, 'reward'
            )

        # Received trained parameters from a worker
        # Update current parameter with damping factors - TODO
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

            logging.info(
                f"<<< Learner Receiving: [Trained Parameters] | Updating parameters")
            
            self.server.agent.load_model(current_parameters)
            self.server.update_agent()

        # unexpected
        else:
            logging.warning(f"Received unexpected data: {type(msg)}")
            return

        # Reply to the request with an up-to-date policy
        send_data(data=PolicyMsg(data=self.server.get_agent_dict()),
                  sock=self.request)


class AsyncLearningNode(ThreadPoolMixIn, socketserver.TCPServer):
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
            update_steps: int = 10,
            batch_size: int = 128,  # Originally 128
            epochs: int = 500,  # Originally 500
            buffer_size: int = 1_000_000,  # Originally 1M
            server_address: Tuple[str, int] = ("0.0.0.0", 4444),
            save_func: Optional[Callable] = None,
            save_freq: Optional[int] = None,
            api_key: str = "",
    ) -> None:
        self.numThreads = 5  # Hardcode for now
        super().__init__(server_address, ThreadedTCPRequestHandler)

        self.update_steps = update_steps
        self.batch_size = batch_size
        self.epochs = epochs

        # Create a replay buffer
        self.buffer_size = buffer_size
        if agent_name == "mountain-car":
            self.replay_buffer = create_configurable(
                "config_files/async_sac_mountaincar/buffer.yaml", NameToSourcePath.buffer
            )
        elif agent_name == "bipedal-walker":
            self.replay_buffer = create_configurable(
                "config_files/async_sac_bipedalwalker/buffer.yaml", NameToSourcePath.buffer
            )
        else:
            print("Invalid Agent Name!")
            exit(1)

        # Initial policy to use
        self.agent = agent
        self.agent_id = 1

        # The bytes of the policy to reply to requests with
        self.updated_agent = {k: v.cpu()
                              for k, v in self.agent.state_dict().items()}

        # A thread-safe policy queue to avoid blocking while learning. This marginally
        # increases off-policy error in order to improve throughput.
        self.agent_queue = queue.Queue(maxsize=1)

        # A queue of buffers that have been received but not yet added to the learner's
        # main replay buffer
        self.buffer_queue = queue.LifoQueue(300)

        self.wandb_logger = WanDBLogger(
            api_key=api_key, project_name="test-project")
        # Save function, called optionally
        self.save_func = save_func

    def get_agent_dict(self) -> Dict[str, Any]:
        """Get the most up-to-date version of the policy without blocking"""
        if not self.agent_queue.empty():
            try:
                self.updated_agent = self.agent_queue.get_nowait()
            except queue.Empty:
                # non-blocking
                pass

        start = time.time()
        task = self.select_task

        if task == Task.TRAIN:
            buffers_to_send = []

            for _ in range(SEND_BATCH):
                batch = self.replay_buffer.sample_batch()
                buffers_to_send.append(batch)

            msg = {
                "policy_id": self.agent_id,
                "policy": self.updated_agent,
                "replay_buffer": buffers_to_send,
                "task": task
            }
        else:
            msg = {
                "policy_id": self.agent_id,
                "policy": self.updated_agent,
                "task": task
            }

        duration = time.time() - start
        if TIMING:
            print(f"Preparation Time = {duration} s")

        logging.info(
            f">>> Learner Sending: [{Task.selection()}] | Param. Ver. = {self.agent_id}")
        return msg

    def update_agent(self) -> None:
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
        while True:
            # Sample data from buffer_queue, and put inside replay buffer
            if not self.buffer_queue.empty() or len(self.replay_buffer) == 0:
                semibuffer = self.buffer_queue.get()

                logging.info(
                    f"--- Learner Processing: Sampled Buffer = {len(semibuffer)} | Replay Buffer = {len(self.replay_buffer)} | Buffer Queue = {self.buffer_queue.qsize()}")

                # Add new data to the primary replay buffer
                self.replay_buffer.store(semibuffer)
            
            time.sleep(0.5)

        # epoch = 0
        # while True:


            # start = time.time()
            # # Learning steps for the policy
            # for _ in range(max(1, min(self.update_steps, len(self.replay_buffer) // self.replay_buffer.batch_size))):
            #     batch = self.replay_buffer.sample_batch()
            #     self.agent.update(data=batch)

            # # Update policy without blocking
            # self.update_agent()
            # duration = time.time() - start
            # if TIMING:
            #     print(f"Update time = {duration}")

            # Optionally save
            # if self.save_func and epoch % self.save_every == 0:
            #     self.save_fn(epoch=epoch, policy=self.get_policy_dict())

            # epoch += 1

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
    