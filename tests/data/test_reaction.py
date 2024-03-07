import pytest

from scaling.data.reaction import Reaction, ReactionStep
from scaling.data.species import Species


class Test_reactionstep:
    def test_init(self):
        reactants = {
            Species("CO2", -1, True, state="NA"): 1,
            Species("H+", -1, False, state="aq"): 1.0,
            Species("e-", -1, False, state="aq"): 1,
        }
        products = {Species("COOH", -1, True, state="NA"): 1}

        reactionstep = ReactionStep(reactants, products)

        # Check if reactants and products are set correctly
        assert reactionstep.reactants == reactants
        assert reactionstep.products == products

    def test_invalid_reactants(self):
        with pytest.raises(TypeError):
            # Pass an invalid type for species
            ReactionStep({1: 2}, {})

        with pytest.raises(TypeError):
            # Pass an invalid type for stoichiometric number
            ReactionStep({Species("CO2", -1, True, state="NA"): "1"}, {})

    def test_invalid_products(self):
        with pytest.raises(TypeError):
            # Pass an invalid type for species
            ReactionStep({}, {1: 2})

        with pytest.raises(TypeError):
            # Pass an invalid type for stoichiometric number
            ReactionStep({}, {Species("COOH", -1, True, state="NA"): "1"})

    def test_negative_stoichiometric_number_warning(self):
        with pytest.warns(UserWarning):
            # Pass a negative stoichiometric number
            ReactionStep({Species("CO2", -1, True, state="NA"): -1}, {})


class Test_reaction:
    def test_init(self):
        reactants = {
            Species("CO2", -1, True, state="NA"): 1,
            Species("H+", -1, False, state="aq"): 1.0,
            Species("e-", -1, False, state="aq"): 1,
        }
        products = {Species("COOH", -1, True, state="NA"): 1}

        reactionstep = ReactionStep(reactants, products)

        Reaction([reactionstep])
