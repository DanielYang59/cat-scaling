from math import isclose

import pytest

from scaling.data.species import Species


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

    def test_str(self):
        species_0 = Species("CO2", -1.0, True, -2.0)
        assert str(species_0) == "*CO2(-1.0, -2.0)"

        species_1 = Species("H2O_g", -2.0, False, -3.0)
        assert str(species_1) == "H2O_g(-2.0, -3.0)"

    def test_invalid_adsorbed_type(self):
        with pytest.raises(TypeError):
            Species("test_species", -1, adsorbed="True")

    def test_invalid_energy_type(self):
        with pytest.raises(TypeError):
            Species("test_species", "hi", adsorbed="True")

    def test_from_str(self):
        species_0 = Species.from_str("*CO2(-1.0, -2.0)")
        assert species_0 == Species("CO2", -1.0, True, -2.0)

        species_1 = Species.from_str("H2O_g(-2.0, -3.0)")
        assert species_1 == Species("H2O_g", -2.0, False, -3.0)

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
