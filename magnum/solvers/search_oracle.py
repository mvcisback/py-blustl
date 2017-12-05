import funcy as fn
import stl
from lenses import bind

from magnum.utils import encode_refuted_rec
from magnum.solvers import smt


@fn.autocurry
def smt_radius_oracle(counter, play, g, r):
    rec = ~encode_refuted_rec(play, r, g.times)
    if rec == stl.BOT:
        rec = stl.TOP

    assert False
    g = bind(g).specs.learned.set(rec)
    return smt.encode_and_run(g, counter_examples=[counter]).feasible


def find_refuted_radius(g, u_star, w_star, num_samples=100,
                        epsilon=0.1, use_smt=True, use_random=False):
    smt_oracle = smt_radius_oracle(counter=w_star, play=u_star, g=g)
    if use_random:
        oracle = check_sat(w_star=w_star, u_star=u_star, g=g,
                           x0=x0, num_samples=num_samples)
    else:
        oracle = smt_oracle

    r_low, r_high, r_mid = 0, 1, 1
    while r_high - r_low > epsilon:
        if oracle(r=r_mid):
            r_low = r_mid
        else:
            r_high = r_mid
        r_mid = (r_low + r_high) / 2.

    return r_high
