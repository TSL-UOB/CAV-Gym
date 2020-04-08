from gym import register

from cavgym import mods

register(
    id='PelicanCrossing-v0',
    entry_point='cavgym.scenarios:PelicanCrossing'
)
