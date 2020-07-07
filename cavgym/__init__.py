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


class AgentType(Enum):
    RANDOM = "random"
    RANDOM_CONSTRAINED = "random-constrained"
