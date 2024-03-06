import pytest

from scaling.data.reaction import Species


class TestSpecies:
    def test_init(self):
        species = Species(
            name="test_species",
            adsorbed=True,
            state="NA",
        )

        assert isinstance(species, Species)
        assert species.name == "test_species"
        assert species.adsorbed is True
        assert species.state == "NA"

    def test_invalid_adsorbed_type(self):
        with pytest.raises(TypeError):
            Species(name="test_species", adsorbed="True", state="NA")

    def test_invalid_state_value(self):
        with pytest.raises(ValueError):
            Species(name="test_species", adsorbed=True, state="invalid_state")
