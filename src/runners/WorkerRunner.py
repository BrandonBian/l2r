from src.runners.base import BaseRunner
from src.config.yamlize import create_configurable, NameToSourcePath, yamlize
from src.constants import DEVICE, Task

from copy import deepcopy
from torch.optim import Adam
import torch
import time


@yamlize
class WorkerRunner(BaseRunner):
    """
    Runner designed for the Worker. All it does is collect data under two scenarios:
      - train, where we include some element of noise
      - test, where we include no such noise.
    """

    def __init__(
        self, agent_config_path: str, buffer_config_path: str, max_episode_length: int
    ):
        super().__init__()
        # Moved initialization of env to run to allow for yamlization of this class.
        # This would allow a common runner for all model-free approaches

        # Initialize runner parameters
        self.agent_config_path = agent_config_path
        self.buffer_config_path = buffer_config_path
        self.max_episode_length = max_episode_length

        # AGENT Declaration
        self.agent = create_configurable(
            self.agent_config_path, NameToSourcePath.agent)

    def run(self, env, agent_params, task):
        """Grab data for system that's needed, and send a buffer accordingly. Note: does a single 'episode'
           which might not be more than a segment in l2r's case.

        Args:
            env (_type_): _description_
            agent (_type_): some agent
            task: eval, collect
        """
        self.agent.load_model(agent_params)
        t = 0
        done = False
        state_encoded, info = env.reset()
        state_encoded = torch.Tensor(state_encoded)

        ep_ret = 0
        self.replay_buffer = create_configurable(
            self.buffer_config_path, NameToSourcePath.buffer
        )
        while not done:
            t += 1

            # Task.eval : deterministic (strictly following the parameters/policy)
            # Task.collect : non-deterministic (willingly explore the space with randomness)
            self.agent.deterministic = (task == Task.EVAL)

            action_obj = self.agent.select_action(state_encoded)
            next_state_encoded, reward, done, terminated, _ = env.step(
                action_obj.action)

            next_state_encoded = torch.Tensor(next_state_encoded)
            state_encoded.to(DEVICE)
            next_state_encoded.to(DEVICE)
            done = done or terminated
            ep_ret += reward

            self.replay_buffer.store(
                {
                    "obs": state_encoded,
                    "act": action_obj,
                    "rew": reward,
                    "next_obs": next_state_encoded,
                    "done": done,
                }
            )
            if done or t == self.max_episode_length:
                self.replay_buffer.finish_path(action_obj)

            state_encoded = next_state_encoded

        info = {'metrics': {}}
        info["metrics"]["reward"] = ep_ret
        return deepcopy(self.replay_buffer), info["metrics"]

    def train(self, agent_params, batches):
        start = time.time()
        self.agent.load_model(agent_params)

        for batch in batches:
            self.agent.update(batch)

        parameters = {k: v.cpu()
                      for k, v in self.agent.state_dict().items()}
        duration = time.time() - start
        
        return {'parameters': parameters, "duration": duration}
