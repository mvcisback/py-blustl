import stl

from magnum.solvers.smt import encode, decode

def test_invertability():
    phi = stl.parse('(G[0, 1](x > 4)) | (~(F[0, 1](y < 5)))')
    psi = stl.utils.discretize(phi, 1)
    smt_phi, store = encode(psi)
    psi2 = decode(smt_phi, store)
    assert psi == psi2
