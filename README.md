![](docs/magnum_vs_stl.jpg)

# YAML Format
## Model
```
model:
 time_horizon: # MPC horizon

 dt: # time steps

 vars:
   state: [x y] # list of state variables
   input: [u] # list of input variables
   env: [w] # list of environment variables

 bounds: # list of key-values, variable: (lower_bound, upper_bound)
   x: (0, 2) # example for variable x
```

## Specification

```
init:
  - stl: x = 0
  - stl: y = 1

sys:
  - name: FooBar # optional name for the constraint
    stl: "F[0,2](x > 5)"
    pri: 1 # optional priority metadata for diagnosis

dyn: # Dynamics, given in slight extension of stl. 
     # Adding primes to variable means previous state.
     # For example: x' is x[t - dt]
  - stl: G[0, 4](-1*u' + dt*1*x' + 5*u = 0)
  - stl: G[0, 4](-1*u' + dt*1*y' + u = 0)
```

See examples/feasible_example.yaml

# Non-Adversarial one off game

```python
import magnum

# Load game scenario
g = magnum.from_yaml("examples/feasible_example.yaml")

# Create spec for game
spec = magnum.one_off_game_to_sl(g)

# Pass spec to MILP oracle and print solution
print(magnum.encode_and_run(spec, g))
```

# Non-Adversarial MPC Usage example

```python
import magnum

# Load game scenario
g = magnum.from_yaml("examples/feasible_example.yaml")

# Create generator receding horizon mpc

controller = magnum.mpc(g)
print(next(controller)[0]) # prints state
print(next(controller)) # prints state, spec
print(controller.send(measurements)) # send measurements and print state, spec
```

# Adversarial MPC Usage example

Coming Soon