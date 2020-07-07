from gym import register

register(
    id='PelicanCrossing-v0',
    entry_point='example.scenarios.pelican_crossing:PelicanCrossingEnv'
)

register(
    id='BusStop-v0',
    entry_point='example.scenarios.bus_stop:BusStopEnv'
)

register(
    id='Crossroads-v0',
    entry_point='example.scenarios.crossroads:CrossroadsEnv'
)

register(
    id='Pedestrians-v0',
    entry_point='example.scenarios.pedestrians:PedestriansEnv'
)