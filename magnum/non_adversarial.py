from magnum.milp import encode_and_run


def predict(phi, g, t):
    return encode_and_run(phi, g)
