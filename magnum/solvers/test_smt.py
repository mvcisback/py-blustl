import stl

from magnum.solvers.smt import encode, decode, encode_and_run

def test_invertability():
    phi = stl.parse('(G[0, 1](x > 4)) | (~(F[0, 1](y < 5)))')
    psi = stl.utils.discretize(phi, 1)
    smt_phi, store = encode(psi)
    psi2 = decode(smt_phi, store)
    assert psi == psi2


def test_feasible_example():
    from magnum.examples.feasible_example import feasible_example as g
    from stl.boolean_eval import pointwise_sat
    res = encode_and_run(g)
    phi = g.spec_as_stl()
    assert pointwise_sat(phi)(res.solution, 0)
    
