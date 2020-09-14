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
  INPUT       read config from INPUT file, or standard input if no INPUT

optional arguments:
  -h, --help  show this help message and exit
```

## Example
```
$ python3 cavgym.py default.json
```
![](demos/pedestrians.gif)
