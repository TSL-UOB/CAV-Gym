# CAV-Gym
A custom OpenAI Gym environment that supports Markov games (joint actions, joint observations, joint rewards) and simple vehicle physics.

## Scenarios

- Bus stop (seed: 16791297828988987546)
![](demos/bus-stop.gif)
- Crossroads (seed: 13390027675813575418)
![](demos/crossroads.gif)
- Pedestrians (seed: 7024298336575781081)
![](demos/pedestrians.gif)
- Pelican crossing (seed: 7391369346644152421)
![](demos/pelican-crossing.gif)

## Usage
```
$ python3 cavgym.py -h                                          
usage: cavgym.py [-h] [-d] [-e N] [-s N] [-t N] [-v]
                 {bus-stop,crossroads,pedestrians,pelican-crossing}
                 {headless,render} ...

positional arguments:
  {bus-stop,crossroads,pedestrians,pelican-crossing}
                        choose scenario to run
  {headless,render}     choose mode to run scenario

optional arguments:
  -h, --help            show this help message and exit
  -d, --debug           print debug information
  -e N, --episodes N    number of episodes (default: 1)
  -s N, --seed N        enable fixed random seed
  -t N, --timesteps N   max number of timesteps per episode (default: 1000)
  -v, --version         show program's version number and exit
```

Additional options are supported in **render** mode, e.g.:
```
$ python3 cavgym.py pelican-crossing render -h
usage: cavgym.py {bus-stop,crossroads,pedestrians,pelican-crossing} render
       [-h] [-k] [-r DIR]

optional arguments:
  -h, --help            show this help message and exit
  -k, --keyboard-agent  run with keyboard-controlled agent
  -r DIR, --record DIR  save video of run to directory
```
