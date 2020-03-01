from gym.envs.registration import register

register(
    id='GridWorldSingle-v0',
    entry_point='gym_GridWorldSingle.envs:GridWorldSingle',
)

register(
    id='GridWorldOrient-v0',
    entry_point='gym_GridWorldSingle.envs:GridWorldOrient',
)