from gym import register

register(
    id='PelicanCrossing-v0',
    entry_point='scenarios.environments.pelican_crossing:PelicanCrossingEnv'
)

register(
    id='BusStop-v0',
    entry_point='scenarios.environments.bus_stop:BusStopEnv'
)

register(
    id='Crossroads-v0',
    entry_point='scenarios.environments.crossroads:CrossroadsEnv'
)

register(
    id='Pedestrians-v0',
    entry_point='scenarios.environments.pedestrians:PedestriansEnv'
)