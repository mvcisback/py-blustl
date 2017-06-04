import stl

from sympy import Symbol

from magnum import game as G
from magnum import io

## Setup the Model

model = G.Model(
    dt=1,
    N=2,
    vars=G.Vars(
        state=(Symbol("x"),),
        input=(Symbol("u"),),
        env=()
    ),
    bounds={
        Symbol("x"): (0, 100),
        Symbol("u"): (0, 1),
    },
    t=0
)

# Setup the specificatoion

context = {
    stl.parse("Init"): stl.parse("x = 0"),
    stl.parse("ReachFive"): stl.parse("F(x > 5)"),
}

spec = G.Specs(
    obj=stl.utils.inline_context(stl.parse("(Init) & (ReachFive)"), context),
    dyn=[stl.parse("G(x + -1*x' + dt*5*u = 0)")],
    learned=[]
)


meta = G.Meta(
    pri={},
    names={},
    dxdu=None,
    dxdw=None,
    drdx=None
)


game = G.Game(spec=spec, model=model, meta=meta)


def main():
    print(io.to_capnp_game(game))


if __name__ == '__main__':
    main()
