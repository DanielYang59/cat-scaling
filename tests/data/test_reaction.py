import pytest

from scaling.data.reaction import ReactionStep, Species


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
            Species(name="test_species", adsorbed="True", state="NA")

    def test_invalid_energy_type(self):
        with pytest.raises(TypeError):
            Species(name="test_species", adsorbed="True", energy="hi")

    def test_invalid_state_value(self):
        with pytest.raises(ValueError):
            Species(name="test_species", adsorbed=True, state="invalid_state")


class TestReactions:
    def test_init(self):
        reactants = {
            Species("CO2", True, state="NA"): 1,
            Species("H+", False, state="aq"): 1.0,
            Species("e-", False, state="aq"): 1,
        }
        products = {Species("COOH", True, state="NA"): 1}

        reaction = ReactionStep(reactants, products)

        # Check if reactants and products are set correctly
        assert reaction.reactants == reactants
        assert reaction.products == products

    def test_invalid_reactants(self):
        with pytest.raises(TypeError):
            # Pass an invalid type for species
            ReactionStep({1: 2}, {})

        with pytest.raises(TypeError):
            # Pass an invalid type for stoichiometric number
            ReactionStep({Species("CO2", True, state="NA"): "1"}, {})

    def test_invalid_products(self):
        with pytest.raises(TypeError):
            # Pass an invalid type for species
            ReactionStep({}, {1: 2})

        with pytest.raises(TypeError):
            # Pass an invalid type for stoichiometric number
            ReactionStep({}, {Species("COOH", True, state="NA"): "1"})

    def test_negative_stoichiometric_number_warning(self):
        with pytest.warns(UserWarning):
            # Pass a negative stoichiometric number
            ReactionStep({Species("CO2", True, state="NA"): -1}, {})
