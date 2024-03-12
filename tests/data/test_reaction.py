# TODO: test reverse Species order in ReactionStep test_from_str

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

        # Test __str__
        assert str(reactionstep) == "1.0*CO2 + 1.0H+_aq + 1.0e-_aq -> 1.0*COOH"

    def test_sepa_stoi_number(self):
        spec_string_0 = " *CO2(0, 0) "
        assert ReactionStep._sepa_stoi_number(spec_string_0) == (
            1.0,
            "*CO2(0, 0)",
        )

        spec_string_1 = " 2H2O_g(-1, 2) "
        assert ReactionStep._sepa_stoi_number(spec_string_1) == (
            2.0,
            "H2O_g(-1, 2)",
        )

    def test_from_str(self):
        react_step = "*A(-1, 0) + 2H2O_g(-2, 3) -> 2*B(-4, 0)"

        # Initialize from Species
        reactants_0 = {
            Species("A", -1.0, True, 0): 1.0,
            Species("H2O", -2.0, False, 3.0, "g"): 2.0,
        }
        products_0 = {
            Species("B", -4.0, True, 0): 2.0,
        }

        assert ReactionStep.from_str(react_step) == ReactionStep(
            reactants=reactants_0, products=products_0
        )

        # Initialize Species from string
        reactants_1 = {
            Species.from_str("*A(-1, 0)"): 1.0,
            Species.from_str("H2O_g(-2, 3)"): 2.0,
        }
        products_1 = {
            Species.from_str("*B(-4, 0)"): 2.0,
        }

        assert ReactionStep.from_str(react_step) == ReactionStep(
            reactants=reactants_1, products=products_1
        )

        # TODO: test reverse Species order

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
