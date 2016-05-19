from collections import namedtuple

Phi = namedtuple("Phi", ["sys", "env", "init"])
SS = namedtuple("StateSpace", ["A", "B"])
Dynamics = namedtuple("Dynamics", ["ss" , "n_vars", "n_sys", "n_env"])
Game = namedtuple("Game", ["phi", "dyn", "width", "dt", "N"])
