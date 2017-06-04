from magnumstl import game as G

model = G.Model(
    dt=1,
    N=2,
    vars=G.Vars(
        state=("x",),
        input=("u",),
        env=()
    ),
    bounds={
        "x": (0, 100),
        "u": (0, 1),
    },
    t=None
)

context = {
    "Init": "x=0",
    "ReachFive": "F(x > 5)",
}

spec = G.Specs(
    obj="(Init) & (ReachFive)",
    dyn=["G(x + -1*x' + dt*5*u = 0)"],
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


if __name__ == '__main__':
    pass
