from itertools import cycle

import stl
import numpy as np

import traces
from functools import reduce
from magnum.solvers.cegis import encode_refuted_rec, combined_solver
from magnum.solvers import smt
from lenses import bind

random_series = lambda g, u , r:traces.TimeSeries(zip(g.scaled_times,
                                        [np.random.uniform(max(0,u[i]-r),
                                                          min(u[i]+r+ 1e-6, 1.))
                                        for i in g.scaled_times]))


def extract_t(controls, w_star, t):
    controls_matrix = np.array([np.array(control_i[t]) for control_i
                                    in controls.values()])
    w_matrix = np.array([np.array(w_i[t]) for w_i in w_star.values()])

    return {'u':controls_matrix, 'w':w_matrix}

def check_sat(g, w_star, u_star, r, x0, num_samples=100):
    A, B, C = g.model.dyn
    dt = g.model.dt
    # Adjust for discrete time
    A = np.eye(len(g.model.vars.state)) + dt * A
    B = dt * B
    C = dt * C

    times = g.scaled_times
    phi = g.spec_as_stl(discretize=False)
    phi_eval = stl.fastboolean_eval.pointwise_sat(phi)

    for _ in range(num_samples):
        controls = {name: random_series(g, u_star[name], r)for name in
                    g.model.vars.input}
        iterate_t = lambda t: extract_t(controls=controls, w_star=w_star, t=t)

        x_ts = [x0]
        for cont in map(iterate_t, times):
            x_ts.append(np.dot(A, x_ts[-1]) + B * cont['u'] + C*cont['w'])
        #x_ts = reduce(lambda x, cont: np.dot(A, x[-1]) + B * cont['u'] +
        #                              C*cont['w'],
        #          map(iterate_t, g.scaled_times), [x0])

        xs = [traces.TimeSeries(zip(times, [x_ts[t][k] for t in g.times]))
              for k in range(len(g.model.vars.state))]
        x = {name: xs[i] for name, i in zip(g.model.vars.state,
                                              range(len(g.model.vars.state)))}
        if phi_eval({**x, **controls, **w_star}, 0):
            return False
    return True

def binary_random_search(g, u_star, w_star, x0, num_samples=100, epsilon =
0.1, use_smt = True):

    r_low, r_high, r_mid = 0, 1, 1

    while True:
        while r_high - r_low > epsilon:
            if r_high <= r_low:
                break
            if check_sat(w_star=w_star, u_star=u_star, r=r_mid, g=g,
                     x0=x0, num_samples=num_samples):
                r_low = r_mid
            else:
                r_high = r_mid
            r_mid = (r_low + r_high) / 2.

        phi_refuted = encode_refuted_rec(u_star, r_high, g.times)
        g = bind(g).specs.learned.set(phi_refuted)

        solve = smt.encode_and_run if use_smt else combined_solver
        res = solve(g, counter_examples=[w_star])
        if res.feasible:
            break

    return r_high













