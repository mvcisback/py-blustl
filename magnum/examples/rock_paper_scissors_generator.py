import stl
import funcy as fn

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


CONTEXT = {
    # Init
    parse("Init"):
    stl.parse('(x = 0) & (y = 0)'),

    # Rules
    parse("PaperBeatsRock"):
    parse("(yPaper) -> (~(xRock))"),
    parse("ScissorsBeatsPaper"):
    parse("(yScissors) -> (~(xPaper))"),
    parse("RockBeatsScissors"):
    parse("(yRock) -> (~(xScissors))"),
    parse("Rules"):
    parse("(PaperBeatsRock) & (ScissorsBeatsPaper) & (RockBeatsScissors)"),
}


def create_intervals(f1, f2, k=0, p1=True, delta=10):
    name = 'x' if p1 else 'y'
    phi = stl.parse(f'({name} >= a?) & ({name} < b?)')

    def intvl_to_spec(itvl):
        return phi.set_params({'a?': itvl[0], 'b?': itvl[1]})

    intervals = ((f1(i), f2(i)) for i in range(k + 1))

    return stl.orf(*map(intvl_to_spec, intervals))


def create_paper(k=0, p1=True, delta=10, eps=0):
    return create_intervals(
        k=k,
        p1=p1,
        delta=delta,
        f1=lambda i: delta * (1 + 6 * i) + eps,
        f2=lambda i: delta * (3 + 6 * i) - eps)


def create_scissors(k=0, p1=True, delta=10, eps=0):
    return create_intervals(
        k=k,
        p1=p1,
        delta=delta,
        f1=lambda i: delta * (3 + 6 * i) + eps,
        f2=lambda i: delta * (5 + 6 * i) - eps)


def create_rock(k=0, p1=True, delta=10, eps=0):
    itvls = create_intervals(
        k=k + 1,
        p1=p1,
        delta=delta,
        f1=lambda i: delta * (-1 + 6 * i) + eps,
        f2=lambda i: delta * (1 + 6 * i) - eps)

    # Hack: to match other rps encoding, we remove first and last one.
    name = 'x' if p1 else 'y'
    start = stl.parse(f'{name} < {delta - eps}')
    end = stl.parse(f'{name} >= {delta*((k+1)*6 - 1) + eps}')
    middle = stl.BOT if k == 0 else stl.orf(*itvls.args[1:-1])
    return start | middle | end


def create_rps_game(n=1, eps=0):
    assert n >= 1

    k = n - 1
    delta = 60 / (6 * n)
    # TODO
    context = {
        'xRock': create_rock(k=k, p1=True, delta=delta, eps=eps),
        'xScissors': create_scissors(k=k, p1=True, delta=delta, eps=eps),
        'xPaper': create_paper(k=k, p1=True, delta=delta, eps=eps),
        'yRock': create_rock(k=k, p1=False, delta=delta),
        'yScissors': create_scissors(k=k, p1=False, delta=delta),
        'yPaper': create_paper(k=k, p1=False, delta=delta),
    }

    context = fn.walk_keys(stl.parse, context)
    context.update(CONTEXT)

    spec = G.Specs(
        obj=stl.parse('G(Rules)', H=H).inline_context(context),
        learned=stl.TOP,
        init=stl.parse("Init").inline_context(context),
    )

    return G.Game(specs=spec, model=MODEL)
