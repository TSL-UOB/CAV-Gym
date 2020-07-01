# CAV-Gym
A custom OpenAI Gym environment that supports Markov games (joint actions, joint observations, joint rewards) and simple vehicle physics.

## Scenarios

### Bus stop
```
$ python3 cavgym.py -s 2701848230254349479 bus-stop render -k
```
![](demos/bus-stop.gif)

### Crossroads
```
$ python3 cavgym.py -s 2655768663906777632 crossroads render -k
```
![](demos/crossroads.gif)

### Pedestrians
```
$ python3 cavgym.py -s 13561506270681953455 pedestrians render -k
```
![](demos/pedestrians.gif)

### Pelican crossing
```
$ python3 cavgym.py -s 1386403555516745859 pelican-crossing render -k
```
![](demos/pelican-crossing.gif)

## Usage
```
$ python3 cavgym.py --help
usage: cavgym.py [-h] [-d] [-e N] [-s N] [-t N] [-v] scenario ...

positional arguments:
  scenario
    bus-stop           three-lane one-way major road with three cars, a
                       cyclist, a bus, and a bus stop
    crossroads         two-lane two-way crossroads road with two cars and a
                       pedestrian
    pedestrians        two-lane one-way major road with a car and a variable
                       number of pedestrians
    pelican-crossing   two-lane two-way major road with two cars, two
                       pedestrians, and a pelican crossing

optional arguments:
  -h, --help           show this help message and exit
  -d, --debug          enable debug mode
  -e N, --episodes N   set number of episodes as N (default: N=1)
  -s N, --seed N       set random seed as N
  -t N, --timesteps N  set max timesteps per episode as N (default: N=1000)
  -v, --version        show program's version number and exit
```

Optional arguments are available for each scenario, in addition to a positional **mode** argument, e.g.:
```
$ python3 cavgym.py pedestrians --help       
usage: cavgym.py pedestrians [-h] [-p N] mode ...

positional arguments:
  mode
    headless            run without rendering
    render              run while rendering to screen

optional arguments:
  -h, --help            show this help message and exit
  -p N, --pedestrians N
                        set number of pedestrians as N (default: N=3)
```

More optional arguments are available in **render** mode, e.g.:
```
$ python3 cavgym.py pedestrians render --help
usage: cavgym.py pedestrians render [-h] [-k] [-r DIR]

optional arguments:
  -h, --help            show this help message and exit
  -k, --keyboard-agent  enable keyboard-control of ego actor
  -r DIR, --record DIR  enable video recording of run to DIR
```
