from blustl.mpc.non_adversarial import non_adversarial_predict
from blustl.mpc.adversarial import cegis

def non_adversarial_mpc(g:Game):
    mpc_specs = non_adversarial_mpc_specs(g)
    phi = next(mpc_specs)
    external_meas = set()
    predict = non_adversarial_predict if len(g.model.vars.env) == 0 else cegis

    while True:
        prediction = predict(phi, g)
        if not prediction.feasible:
            return prediction
        predicted_meas = prediction.solution[g.model.N]
        phi = mpc_specs.send(predicted_meas | external_meas)
        external_meas = yield predicted_meas
