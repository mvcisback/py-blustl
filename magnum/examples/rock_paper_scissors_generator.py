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

def create_paper_scissors(f1, f2, k=0, p1=True, delta=10):
    name = 'x' if p1 else 'y'
    phi = stl.parse(f'({name} >= a?) & ({name} < b?)')
    
    def intvl_to_spec(itvl):
        return phi.set_params({'a?': itvl[0], 'b?': itvl[1]})

    intervals = ((f1(i), f2(i)) for i in range(k+1))
    
    return stl.andf(*map(intvl_to_spec, intervals))


def create_paper(k=0, p1=True, delta=10):
    return create_paper_scissors(k=k, p1=p1, delta=delta, 
                                 f1=lambda i: delta*(1 + 6*i),
                                 f2=lambda i: delta*(3 + 6*i))


def create_rps_game(n=1, spock=0):
    assert n >= 1
    
    # TODO
    context = CONTEXT

    spec = G.Specs(
        obj=stl.parse('G(Rules)', H=H).inline_context(context),
        learned=stl.TOP,
        init=stl.parse("Init").inline_context(context),
    )

    rps = G.Game(specs=spec, model=model)
