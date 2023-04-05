# l2r - Distributed RL with Optimization Thrust
## Branches
- [main]
- [distributed-walker] (formerly `l2r-benchmarks -> distributed-mcar branch`)
- [distributed-l2r] (formerly `l2r-benchmarks -> distributed-worker branch`)
- [sequential-l2r] (formerly `l2r-lab -> phoebe branch`)


## Bug fixes
### l2r-deployment.yaml
1. Prepended `apt-get update` for worker-pods' command
2. Replaced `worker.py` and `learner.py` with `distributedworker.py` and `distributedserver.py` respectively for worker-pods and learner-pod
3. Added `tensorboardX` to the installation command for learner-pod

### [distributed-walker] branch
1. Removed `entity="learn2race"` from `src/loggers/WanDBLogger.py`

### [distributed-l2r] branch
1. Moved `distributedworker.py` and `distributedlearner.py` from `./scripts/` to `root`
2. `./src/config/schema.py` missing fields from `config_files/async_sac/agent.yaml`