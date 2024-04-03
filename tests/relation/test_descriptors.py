import pytest

from cat_scaling.relation import Descriptors


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

    def test_groups_invalid(self):
        # Test invalid groups dtype
        with pytest.raises(TypeError, match="Expect groups as dict"):
            Descriptors("Groups", method="traditional")

        # Test invalid groups key dtype
        with pytest.raises(
            TypeError, match="Keys in groups dictionary must be strings"
        ):
            Descriptors(
                {
                    0: ["*COOH", "*CHO", "*CH2O"],
                    "*OH": ["*OCH3", "*O", "*CH2O"],
                },
                method="traditional",
            )

        # Test invalid groups value dtype
        with pytest.raises(
            TypeError, match="Group members must be lists of strings or None"
        ):
            Descriptors(
                {
                    "*CO": "*COOH",
                    "*OH": [
                        "*OCH3",
                    ],
                },
                method="traditional",
            )

        # Test group members overlap
        groups = {
            "*CO": ["*COOH", "*CHO", "*CH2O"],
            "*OH": ["*OCH3", "*O", "*CH2O"],
        }

        with pytest.warns(
            UserWarning, match="Descriptor group members overlap."
        ):
            Descriptors(groups, method="traditional")

    def test_property_method(self):
        with pytest.raises(ValueError, match="Invalid method"):
            groups = {
                "*CO": ["*COOH", "*CHO", "*CH2O"],
                "*OH": ["*OCH3", "*O"],
            }

            Descriptors(groups, method="invalid")
