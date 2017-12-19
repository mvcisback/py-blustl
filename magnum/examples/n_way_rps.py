import stl

from magnum import game as G

import numpy as np

H = 1

MODEL = G.Model(
    dt=1,
    vars=G.Vars(state=("x", "y"), input=("u", ), env=("w", )),
    dyn=G.Dynamics(
        A=np.array([[0, 0], [0, 0]]),
        B=np.array([60, 0]).reshape((2, 1)),
        C=np.array([0, 60]).reshape((2, 1)),
    ))


def parse(x):
    return stl.parse(x, H=H)


def create_intervals(f1, f2, k=0, p1=True, delta=10):
    name = 'x' if p1 else 'y'
    phi = parse(f'({name} >= a?) & ({name} < b?)')

    def intvl_to_spec(itvl):
        return phi.set_params({'a?': itvl[0], 'b?': itvl[1]})

    intervals = ((f1(i), f2(i)) for i in range(k + 1))

    return map(intvl_to_spec, intervals)


def gen_moves(k=0, p1=True, delta=10, eps=0):
    return create_intervals(
        k=k,
        p1=p1,
        delta=delta,
        f1=lambda i: delta * (i) + eps - 60,
        f2=lambda i: delta * (i + 1) - eps - 60)


def create_game(n=2, eps=0):
    assert n >= 1

    k = n - 1
    delta = 2*60 / n + 1e-2  # add padding to make last interval closed.
    # TODO
    p1_moves = gen_moves(k, p1=True, delta=delta, eps=eps)
    p2_moves = gen_moves(k, p1=False, delta=delta)

    def create_rule(i):
        return parse(f'(yMove{(i + 1) % n}) -> (~(xMove{i}))')

    context = {
        parse(f'xMove{i}'): move for i, move in enumerate(p1_moves)}
    context.update(
        {parse(f'yMove{i}'): move for i, move in enumerate(p2_moves)})
    context.update({
        parse("Init"): parse('(x = 0) & (y = 0)'),
        parse('Rules'): stl.andf(*map(create_rule, range(n))),
    })

    spec = G.Specs(
        obj=parse('G(Rules)').inline_context(context),
        learned=stl.TOP,
        init=parse("Init").inline_context(context),
    )

    return G.Game(specs=spec, model=MODEL)
