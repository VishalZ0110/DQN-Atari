import gymnasium as gym
import ale_py

gym.register_envs(ale_py)


def make_env(env_name, frameskip=4, seed=None, render_mode=None):
    """
    Factory function for creating Atari environments.
    """
    def thunk():
        env = gym.make(env_name, frameskip=frameskip, render_mode=render_mode)
        if seed is not None:
            env.action_space.seed(seed)
        return env
    return thunk
