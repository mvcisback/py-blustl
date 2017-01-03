from itertools import chain, repeat

from blustl.mpc import non_adversarial
from blustl.mpc.adversarial import cegis
from blustl.game import Game

def mpc(g:Game):
    mpc_specs = non_adversarial.specs(g)
    phi = next(mpc_specs)
    external_meas = set()
    predict = non_adversarial.predict if len(g.model.vars.env) == 0 else cegis
    H = 2*g.model.N
    for t in chain(range(H), repeat(H)):
        prediction = predict(phi, g)
        if not prediction.feasible:
            return prediction
        predicted_meas = prediction.solution.get(t, set())
        phi = mpc_specs.send(predicted_meas | external_meas)
        external_meas = yield predicted_meas
        if external_meas is None:
            external_meas = set()
