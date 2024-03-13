# TODO: add unit test for property: method

from scaling.relation import Descriptors


class Test_descriptors:
    def test_init(self):
        # Initialize "traditional" descriptors
        groups_trad = {
            "*CO": ["*COOH", "*CHO", "*CH2O"],  # C-centered group
            "*OH": ["*OCH3", "*O"],  # O-Centered group
        }

        des_trad = Descriptors(groups_trad, method="traditional")

        assert des_trad.descriptors == ["*CO", "*OH"]
        assert len(des_trad) == 2

        # Initialize "adaptive" descriptors
        groups_adap = {
            "*CO": None,
            "*OH": None,
        }

        des_adap = Descriptors(groups_adap, method="traditional")

        assert des_adap.descriptors == ["*CO", "*OH"]
        assert len(des_adap) == 2
