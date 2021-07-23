from gym.envs.registration import register

register(
    id="mimocontrol-v0", entry_point="laser_cbc.envs:LaserControllerEnv",
)

register(
    id="mimocontrol-v1", entry_point="laser_cbc.envs:LaserControllerRLEnv",
)

__all__ = ["envs"]

