from distrib_l2r.asynchron.learner import AsyncLearningNode
from src.config.yamlize import NameToSourcePath, create_configurable
#from tianshou.policy import SACPolicy
#from tianshou.utils.net.common import Net
#from tianshou.utils.net.continuous import ActorProb, Critic
import torch
from torch import nn
import threading
import numpy as np
import time
import sys
import os

if __name__ == "__main__":

    agent_name = os.getenv("AGENT_NAME")
    print(f"Learner Initialized - {agent_name}")

    if agent_name == "bipedal-walker":
        learner = AsyncLearningNode(
                agent=create_configurable(
                    "config_files/async_sac_bipedalwalker/agent.yaml", NameToSourcePath.agent
                ),
                api_key=sys.argv[1],
            )
    elif agent_name == "mountain-car":
        learner = AsyncLearningNode(
                agent=create_configurable(
                    "config_files/async_sac_mountaincar/agent.yaml", NameToSourcePath.agent
                ),
                api_key=sys.argv[1],
            )
    else:
        print("Invalid Agent Name!")
        exit(1)

    server_thread = threading.Thread(target=learner.serve_forever)
    server_thread.start()
    print("Learning?")
