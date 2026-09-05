"""Attach the unchanged official RSL critic to peer-final bootstrap."""
from __future__ import annotations

from .semantic_training import construct_semantic_runner, train_semantic
from .rl_library_wrapper import seed_training_rngs


def construct_semantic_vector_runner(env, *, seed: int, device: str):
    seed_training_rngs(seed)
    runner, config = construct_semantic_runner(env,seed=seed,device=device)
    env.bind_final_value_function(runner.alg.critic,gamma=runner.alg.gamma)
    return runner,config


def train_semantic_vector(runner, env, *, decisions: int, **kwargs):
    """Keep N=8 and exact total budgets; no silent 1024-transition tail overrun."""
    batch = int(runner.cfg["num_steps_per_env"])*env.num_envs
    if env.num_envs!=8 or type(decisions) is not int or decisions<=0 or decisions%batch:
        raise ValueError(f"semantic vector v1 requires N=8 and an exact multiple of {batch} total decisions")
    if env._final_value_function is None:
        raise RuntimeError("construct_semantic_vector_runner must bind final critic")
    return train_semantic(runner,env,decisions=decisions,**kwargs)
