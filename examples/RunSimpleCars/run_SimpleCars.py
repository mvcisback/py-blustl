# Simple Cars example
import blustl
from blustl.milp import sl_to_milp

# Load game scenario
g = blustl.from_yaml("SimpleCars.yaml")

# Create spec for game
spec = blustl.one_off_game_to_sl(g)

# Pass spec to MILP oracle and print solution
model,store = sl_to_milp(spec,g)
result = blustl.encode_and_run(spec, g)
print(result.solution)

