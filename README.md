# CAV-Gym
A library for building 2D road environments based on a multi-agent variant of OpenAI Gym (joint actions, joint observations, joint rewards).
Four environments and a collection of agents are included as examples.

## Dependencies
Requires ```Python >= 3.7```.

## Usage
```
$ python3 cavgym.py --help
usage: cavgym.py [-h] [INPUT]

positional arguments:
  INPUT       read config from INPUT file, or from stdin if no file is provided

optional arguments:
  -h, --help  show this help message and exit
```

## Example
```
$ python3 cavgym.py config.json
INFO    302     config.py:setup             seed=0
INFO    308     config.py:setup             bodies=[Car, SpawnPedestrian]
INFO    308     config.py:setup             agents=[NoopAgent, RandomConstrainedAgent]
INFO    308     config.py:setup             ego=(Car, NoopAgent)
INFO    572     simulation.py:run           episode 1 terminated after 675 timestep(s) in 264 ms (*42.62 real-time), uninteresting test
INFO    572     simulation.py:run           run completed after 1 episode(s) and 675 timestep(s) in 264 ms (*42.59 real-time), no interesting test(s)
```
![](demos/pedestrians.gif)
