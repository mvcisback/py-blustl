from nose2.tools import params
import unittest

from blustl import repair, stl_parser, stl
"""
class TestSTLParser(unittest.TestCase):
    @params(
        ('F[0, 1](x1 > 2)', 'F[0, 1](x1 > 2)'),
        ('G[0, 5]F[0, 1](x1 > 2)', 'F[0, ?](x1 > 2)'),
        ('F[1, 5]G[0, 1](x1 > 2)', 'G[1, ?]F[?, ?](x1 > 2)'),
        ('G[0, 5](x1 > 2)', 'F[0, ?]G[0, 5](x1 > 2)'),
    )
    def test_temporal_weaken(self, phi, phi2):
        phi = stl_parser.parse_stl(phi)
        phi2 = stl_parser.parse_stl(phi2)
        phi = repair.temporal_weaken(phi)
        self.assertEqual(phi, phi2)

    @params(
        ('F[0, 1](x1 > 2)', 'G[?, ?]F[0, 1](x1 > 2)'),
        ('G[0, 5]F[0, 1](x1 > 2)', 'F[0, 5]G[?, ?](x1 > 2)'),
        ('F[1, 5]G[0, 1](x1 > 2)', 'G[1, 5](x1 > 2)'),
        ('G[0, 1](x1 > 2)', 'G[0, 1](x1 > 2)'),
    )
    def test_temporal_weaken(self, phi, phi2):
        phi = stl_parser.parse_stl(phi)
        phi2 = stl_parser.parse_stl(phi2)
        phi = repair.temporal_strengthen(phi)
        self.assertEqual(phi, phi2)

    def structure_test(self, n, change_structure, kind):
        phi = stl_parser.parse_stl("G[?, ?](x2 < 4)")
        r = list(change_structure(phi, n))
        self.assertEqual(len(r), 2*n)
        for phi2 in r:
            self.assertEqual(phi2.args[0], phi)
            self.assertEqual(type(phi2.args[1]), kind)
        

    @params(1, 3)
    def test_structure(self, n):
        self.structure_test(n, repair.weaken_structure, stl.F)
        self.structure_test(n, repair.strengthen_structure, stl.G)


    def test_all_repairs_smoke_tests(self):
        phi = stl_parser.parse_stl("G[?, ?](x2 < 4)")
        n = 3
        l = n*2 + 1
        self.assertEqual(
            len(list(repair.all_repairs(phi, n, strengthen=False))), l)
        self.assertEqual(
            len(list(repair.all_repairs(phi, n, strengthen=True))), l)
"""
