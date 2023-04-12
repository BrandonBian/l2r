import socket
from distrib_l2r.asynchron.worker import AsnycWorker
from src.config.yamlize import create_configurable

# from src.utils.envwrapper_aicrowd import EnvContainer
#from tianshou.policy import SACPolicy
#from tianshou.utils.net.common import Net
#from tianshou.utils.net.continuous import ActorProb, Critic
import torch
from torch import nn
import numpy as np
import time
import os

if __name__ == "__main__":

    agent_name = os.getenv("AGENT_NAME")
    print(f"Worker Initialized - {agent_name}")

    if agent_name == "bipedal-walker":
        learner_ip = socket.gethostbyname("walker-learner-service")
        learner_address = (learner_ip, 4444)
        worker = AsnycWorker(learner_address=learner_address)
    else if agent_name == "mountain-car":
        learner_ip = socket.gethostbyname("mcar-learner-service")
        learner_address = (learner_ip, 4444)
        worker = AsnycWorker(learner_address=learner_address)

    worker.work()
