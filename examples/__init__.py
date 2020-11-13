from gym import register

register(
    id='PelicanCrossing-v0',
    entry_point='examples.environments.pelican_crossing:PelicanCrossingEnv'
)

register(
    id='BusStop-v0',
    entry_point='examples.environments.bus_stop:BusStopEnv'
)

register(
    id='Crossroads-v0',
    entry_point='examples.environments.crossroads:CrossroadsEnv'
)

register(
    id='Pedestrians-v0',
    entry_point='examples.environments.pedestrians:PedestriansEnv'
)
