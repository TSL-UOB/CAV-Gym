from enum import Enum

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


class Scenario(Enum):
    BUS_STOP = "bus-stop"
    CROSSROADS = "crossroads"
    PEDESTRIANS = "pedestrians"
    PELICAN_CROSSING = "pelican-crossing"

    def __str__(self):
        return self.value


CONFIG = {
    Scenario.PELICAN_CROSSING: ('PelicanCrossing', 'cavgym.scenarios.pelican_crossing:PelicanCrossingEnv'),
    Scenario.BUS_STOP: ('BusStop', 'cavgym.scenarios.bus_stop:BusStopEnv'),
    Scenario.CROSSROADS: ('Crossroads', 'cavgym.scenarios.crossroads:CrossroadsEnv'),
    Scenario.PEDESTRIANS: ('Pedestrians', 'cavgym.scenarios.pedestrians:PedestriansEnv')
}


def register_seeded(scenario, seed, version=0):
    scenario_id, entry_point = CONFIG[scenario]
    register_id = f'{scenario_id}{seed}-v{version}'
    register(
        id=register_id,
        entry_point=entry_point,
        kwargs={'seed': seed}
    )
    return register_id
