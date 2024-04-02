from math import isclose

import pytest

from cat_scaling.data.species import Species


class Test_species:
    def test_init(self):
        species = Species(
            name="test_species",
            adsorbed=True,
            energy=-1,
        )

        assert isinstance(species, Species)
        assert species.name == "test_species"
        assert species.adsorbed is True
        assert isclose(species.energy, -1.0)

    def test_eq(self):
        species_0 = Species("CO2", -1.0, True, -2.0)
        assert species_0 != "Invalid type"

    def test_str(self):
        species_0 = Species("CO2", -1.0, True, -2.0)
        assert str(species_0) == "*CO2(-1.0, -2.0)"

        species_1 = Species("H2O_g", -2.0, False, -3.0)
        assert str(species_1) == "H2O_g(-2.0, -3.0)"

    def test_invalid_adsorbed_type(self):
        with pytest.raises(TypeError, match="Adsorbed should be boolean"):
            Species("test_species", -1, adsorbed="True")

    def test_invalid_energy_type(self):
        with pytest.raises(TypeError, match="Energy should be float"):
            Species("test_species", "hi", adsorbed="True")

    def test_invalid_correction_dtype(self):
        with pytest.raises(TypeError, match="Correction should be float"):
            Species("test_species", -1, adsorbed=True, correction="wrong_type")

    def test_from_str(self):
        species_0 = Species.from_str("*CO2(-1.0, -2.0)")
        assert species_0 == Species("CO2", -1.0, True, -2.0)

        species_1 = Species.from_str("H2O_g(-2.0, -3.0)")
        assert species_1 == Species("H2O_g", -2.0, False, -3.0)

    def test_from_str_invalid(self):
        with pytest.raises(TypeError, match="Expect type str"):
            Species.from_str([1, 2])  # expect a str

        with pytest.raises(
            ValueError, match="Invalid format for energy and correction"
        ):
            Species.from_str("H2O_g(-2.0, -3.0, invalid)")

    def test_from_dict(self):
        species_0 = Species.from_dict(
            {
                "name": "CO2",
                "energy": -2.5,
                "adsorbed": True,
                "correction": 0.1,
            }
        )
        assert species_0 == Species("CO2", -2.5, True, 0.1)

        species_1 = Species.from_dict(
            {
                "name": "H2O",
                "energy": -2,
                "adsorbed": False,
            }
        )
        assert species_1 == Species("H2O", -2, False, 0.0)

    def test_from_dict_invalid(self):
        with pytest.raises(TypeError, match="Expect a dict"):
            Species.from_dict("invalid type")

        with pytest.raises(ValueError, match="Missing required arg"):
            Species.from_dict(
                {
                    "name": "H2O",
                    "energy": -2,
                    # "adsorbed": False,
                }
            )
