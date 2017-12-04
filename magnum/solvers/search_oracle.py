from itertools import cycle

import stl
import numpy as np

import traces
from functools import reduce

random_series = lambda g: traces.TimeSeries(zip(g.scaled_times,
                                        np.random.random(len(g.scaled_times))))


def extract_t(controls, w_star, t):
    controls_matrix = np.array([np.array(control_i).T[1] for control_i
                                    in controls.values()])
    w_matrix = np.array([np.array(w_i).T[1] for w_i in w_star.values()])

    return {'u':controls_matrix[:, [t]], 'w':w_matrix.T[:,[t]]}

def check_sat(w_star, u_star, r, g, num_samples=100):
    A, B, C = g.model.dyn
    dt = g.model.dt
    # Adjust for discrete time
    A = np.eye(len(g.model.vars.state)) + dt * A
    B = dt * B
    C = dt * C

    times = g.scaled_times
    phi = g.spec_as_stl(dizcretize=False)
    phi_eval = stl.boolean_eval.pointwise_sat(phi)

    for _ in range(num_samples):
        controls = {name: random_series(g) * r + u_star[name] for name in
                g.model.vars.input}
        iterate_t = lambda t: extract_t(controls=controls, w_star=w_star, t=t)
        x_ts = reduce(lambda x, cont: A * x[-1] + B * cont['u'] + C(cont['w']),
                  map(iterate_t, g.times), [x0])

        x_ts = traces.TimeSeries(zip(times, x_ts))
        x = {name: x_ts[i] for name, i in zip(g.model.vars.state,
                                              len(g.model.vars.state))}
        if not phi_eval({**x, **controls, **w_star}, 0):
            return False

    return True





def binary_random_search(g, u_star, w_star, x0, num_samples=100, epsilon = 0.1):


    r_low, r_high, r_mid = 0, 1, 1

    while r_high - r_low > epsilon:
        if r_high <= r_low:
            break
        if check_sat(w_star, u_star, r_mid, g, num_samples):
            r_low = r_mid
        else:
            r_high = r_mid
        r_mid = (r_low + r_high) / 2.

    return r_mid













