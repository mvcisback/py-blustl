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

    assert phi0 | phi1 | phi2 == psi


def test_create_scissors():
    psi = rpsg.create_scissors(k=0)
    phi0 = stl.parse('(x >= 30) & (x < 50)')

    assert phi0 == psi

    psi = rpsg.create_scissors(k=2, p1=False)
    phi0 = stl.parse('(y >= 30) & (y < 50)')
    phi1 = stl.parse('(y >= 90) & (y < 110)')
    phi2 = stl.parse('(y >= 150) & (y < 170)')

    assert phi0 | phi1 | phi2 == psi


def test_create_rock():
    psi = rpsg.create_rock(k=0)
    phi0 = stl.parse('(x < 10)')
    phi1 = stl.parse('(x >= 50)')

    assert phi0 | phi1 == psi

    psi = rpsg.create_rock(k=2, p1=False)
    phi0 = stl.parse('(y < 10)')
    phi1 = stl.parse('(y >= 50) & (y < 70)')
    phi2 = stl.parse('(y >= 110) & (y < 130)')
    phi3 = stl.parse('(y >= 170)')

    assert phi0 | phi1 | phi2 | phi3 == psi


def test_create_rps_game():
    from magnum.examples.rock_paper_scissors import rps as g
    g2 = rpsg.create_rps_game(n=1, eps=0)
    assert g.specs == g2.specs
