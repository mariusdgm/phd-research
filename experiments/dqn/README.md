### TODO: 

- implement the bm error formula in the agent as it is used in the src code - check, but could not just bring the function because we have 2 models
- modify epsilon greedy to reach exploitation after 33% - done
- add a parameter for skewness (entropy) for the replay buffer to check distribution - done


### Commands
```
liftoff-prepare configs --runs-no 40 --do
```

```
liftoff experiment_dqn.py .\results\2024Jun05-130920_configs --procs-no 8
```

## To install required env library:
```
pip install rl-envs-forge==3.7.0
```