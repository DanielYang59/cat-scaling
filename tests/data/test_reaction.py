import pytest

from cat_scaling.data.reaction import Reaction, ReactionStep
from cat_scaling.data.species import Species

# Dummy energies for test
energy_dict = {
    "A": (-1, 0.5),
    "B": (-2, 1.0),
    "C": (-3, 1.5),
    "H2O_g": (-4, 2.0),
    "H2_g": (-5, 2.5),
    "CO2": (-6, 3.0),
    "COOH": (-7, 3.5),
    "CO": (-8, 4.0),
    "H+": (-9, 4.5),
    "e-": (0, 0),
}


@pytest.mark.filterwarnings("ignore:Non-negative energy found")
class Test_reactionstep:
    def test_init(self):
        reactants = {
            Species("CO2", -6, True, 3): 1,
            Species("H+", -9, False, 4.5): 1.0,
            Species("e-", 0, False, 0): 1,
        }
        products = {Species("COOH", -7, True, 3.5): 1}

        reactionstep = ReactionStep(reactants, products)

        # Check if reactants and products are set correctly
        assert reactionstep.reactants == reactants
        assert reactionstep.products == products

        # Test __str__
        assert str(reactionstep) == "1.0*CO2 + 1.0H+ + 1.0e- -> 1.0*COOH"

    def test_eq(self):
        react_step = "*A + 2H2O_g -> 2*B"

        # Initialize from Species
        reactants = {
            Species("A", -1, True, 0.5): 1,
            Species("H2O_g", -4, False, 2.0): 2,
        }

        products = {
            Species("B", -2, True, 1.0): 2,
        }

        assert ReactionStep.from_str(react_step, energy_dict) == ReactionStep(
            reactants=reactants, products=products
        )

    def test_sepa_stoi_number(self):
        spec_string_0 = " *CO2(-6, 3) "
        assert ReactionStep._sepa_stoi_number(spec_string_0) == (
            1.0,
            "*CO2(-6, 3)",
        )

        spec_string_1 = " 2H2O_g(-4, 2) "
        assert ReactionStep._sepa_stoi_number(spec_string_1) == (
            2.0,
            "H2O_g(-4, 2)",
        )

    def test_from_str(self):
        react_step = "*A + 2H2O_g -> 2*B"

        # Initialize from Species
        reactants_0 = {
            Species("A", -1.0, True, 0.5): 1.0,
            Species("H2O_g", -4.0, False, 2.0): 2.0,
        }
        products_0 = {
            Species("B", -2.0, True, 1): 2.0,
        }

        assert ReactionStep.from_str(react_step, energy_dict) == ReactionStep(
            reactants=reactants_0, products=products_0
        )

        # Initialize Species from string
        reactants_1 = {
            Species.from_str("*A(-1, 0.5)"): 1.0,
            Species.from_str("H2O_g(-4, 2.0)"): 2.0,
        }
        products_1 = {
            Species.from_str("*B(-2, 1)"): 2.0,
        }

        assert ReactionStep.from_str(react_step, energy_dict) == ReactionStep(
            reactants=reactants_1, products=products_1
        )

    def test_invalid_reactants(self):
        with pytest.raises(TypeError):
            # Pass an invalid type for species
            ReactionStep({1: 2}, {})

        with pytest.raises(TypeError):
            # Pass an invalid type for stoichiometric number
            ReactionStep({Species("CO2", -6, True, 3): "1"}, {})

    def test_invalid_products(self):
        with pytest.raises(TypeError):
            # Pass an invalid type for species
            ReactionStep({}, {1: 2})

        with pytest.raises(TypeError):
            # Pass an invalid type for stoichiometric number
            ReactionStep({}, {Species("COOH", -7, True, 3.5): "1"})

    def test_negative_stoichiometric_number_warning(self):
        with pytest.warns(UserWarning):
            # Pass a negative stoichiometric number
            ReactionStep({Species("CO2", -6, True, 3): -1}, {})


@pytest.mark.filterwarnings("ignore:Non-negative energy found")
class Test_reaction:
    def test_init(self):
        reactants = {
            Species("CO2", -6, True, 3.0): 1,
            Species("H+", -9, False, 4.5): 1.0,
            Species("e-", 0, False): 1,
        }
        products = {Species("COOH", -7, True, 3.5): 1}

        reactionstep = ReactionStep(reactants, products)

        _reaction = Reaction([reactionstep])

        assert isinstance(_reaction, Reaction)

    def test_from_str(self):
        test_str = """
        *A + 2H2O_g -> 2*B
        *B -> *C + H2_g
        """
        reaction = Reaction.from_str(test_str, energy_dict)
        assert len(reaction) == 2
        assert reaction[0] == ReactionStep.from_str(
            "*A + 2H2O_g -> 2*B", energy_dict
        )
        assert reaction[1] == ReactionStep.from_str(
            "*B -> *C + H2_g", energy_dict
        )
