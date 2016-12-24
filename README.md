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

# Usage example

```python
import blustl

g = blustl.from_yaml("examples/feasible_example.yaml") # Game Object
phi = blustl.mpc_game_to_sl(g) # Time discretized STL
res = blustl.encode_and_run(phi) # MPC Result
```