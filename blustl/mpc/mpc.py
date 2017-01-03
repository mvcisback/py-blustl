from blustl.mpc import non_adversarial
from blustl.mpc.adversarial import cegis
from blustl.game import Game

def non_adversarial_mpc(g:Game):
    mpc_specs = non_adversarial.specs(g)
    phi = next(mpc_specs)
    external_meas = set()
    predict = non_adversarial.predict if len(g.model.vars.env) == 0 else cegis

    while True:
        prediction = predict(phi, g)
        if not prediction.feasible:
            return prediction
        predicted_meas = prediction.solution.get(0, set())
        phi = mpc_specs.send(predicted_meas | external_meas)
        external_meas = yield predicted_meas
        if external_meas is None:
            external_meas = set()
