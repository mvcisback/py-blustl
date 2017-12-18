import stl

from magnum.examples import rock_paper_scissors_generator as rpsg

def test_create_paper():
    psi = rpsg.create_paper(k=0)
    phi0 = stl.parse('(x >= 10) & (x < 30)')
    
    assert phi0 == psi

    psi = rpsg.create_paper(k=2, p1=False)
    phi0 = stl.parse('(y >= 10) & (y < 30)')
    phi1 = stl.parse('(y >= 70) & (y < 90)')
    phi2 = stl.parse('(y >= 130) & (y < 150)')
    
    assert phi0 & phi1 & phi2 == psi
