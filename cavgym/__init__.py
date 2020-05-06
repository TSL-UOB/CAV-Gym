from gym import register

from cavgym import mods

register(
    id='PelicanCrossing-v0',
    entry_point='cavgym.scenarios.pelican_crossing:PelicanCrossingEnv'
)

register(
    id='BusStop-v0',
    entry_point='cavgym.scenarios.bus_stop:BusStopEnv'
)

register(
    id='Crossroads-v0',
    entry_point='cavgym.scenarios.crossroads:CrossroadsEnv'
)

register(
    id='Pedestrians-v0',
    entry_point='cavgym.scenarios.pedestrians:PedestriansEnv'
)
