import pytest

from scaling.data.species import Species


class Test_species:
    def test_init(self):
        species = Species(
            name="test_species",
            adsorbed=True,
            energy=-1,
            state="NA",
        )

        assert isinstance(species, Species)
        assert species.name == "test_species"
        assert species.adsorbed is True
        assert species.energy == -1.0
        assert species.state == "NA"

    def test_invalid_adsorbed_type(self):
        with pytest.raises(TypeError):
            Species("test_species", -1, adsorbed="True", state="NA")

    def test_invalid_energy_type(self):
        with pytest.raises(TypeError):
            Species("test_species", "hi", adsorbed="True")

    def test_invalid_state_value(self):
        with pytest.raises(ValueError):
            Species("test_species", -1, adsorbed=True, state="invalid_state")
