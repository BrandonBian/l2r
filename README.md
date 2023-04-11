# l2r - Distributed RL with Optimization Thrust
## Branches
- [main]
- [distributed-walker] (formerly `l2r-benchmarks -> distributed-mcar branch`)
- [distributed-l2r] (formerly `l2r-benchmarks -> distributed-worker branch`)
- [sequential-l2r] (formerly `l2r-lab -> phoebe branch`)
- [sequential-walker]

## Bug fixes
### l2r-distributed.yaml
1. Prepended `apt-get update` for worker-pods' command
2. Replaced `worker.py` and `learner.py` with `distributedworker.py` and `distributedserver.py` respectively for worker-pods and learner-pod
3. Added `tensorboardX` to the installation command for learner-pod

### [distributed-walker] branch
1. Removed `entity="learn2race"` from `src/loggers/WanDBLogger.py`

### [distributed-l2r] branch
1. Moved `distributedworker.py` and `distributedlearner.py` from `./scripts/` to `root`
2. `./src/config/schema.py` missing fields from `config_files/async_sac/agent.yaml`

### [distributed-mcar] branch (mountain car continuous)
1. Change the default action space from `self.action_space = Box(-1, 1, (4,))` to `self.action_space = Box(-1, 1, (self.actor_critic.action_dim,))` in `src/agents/SACAgent.py`
2. Add `self.action_dim = action_dim` to `class ActorCritic(nn.Module)` in `src/networks/critic.py`, so that the above command can work
3. Add the checking of action being a scalar in `select_action()` in `src/agents/SACAgents.py`
```python
def select_action(self, obs):
    ...
    a = self.actor_critic.act(obs.to(DEVICE), self.deterministic)
    if a.shape == ():
        # In case a in a scalar
        a = np.array([a])
    action_obj.action = a
    ...
```
