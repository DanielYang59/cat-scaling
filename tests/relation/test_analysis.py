# TODO: need to pre-define a set of Relation/Reaction (or such) for tests
# as currently definition of Relation/Reaction is repeated

import numpy as np
import pandas as pd
import pytest

from cat_scaling.data import Eads
from cat_scaling.data.reaction import Reaction, ReactionStep, Species
from cat_scaling.relation.analysis import AdsorbToDeltaE
from cat_scaling.relation.builder import Builder
from cat_scaling.relation.descriptors import Descriptors
from cat_scaling.relation.relation import DeltaERelation
from cat_scaling.utils import PROJECT_ROOT


class Test_AdsorbToDeltaE:
    @pytest.fixture
    def eads_relation(self):
        # Load data
        test_data_csv = (
            PROJECT_ROOT / "tests" / "relation" / "relation_data.csv"
        )
        test_df = pd.read_csv(test_data_csv, index_col=[0], header=[0])
        eads = Eads(test_df)
        descriptors = Descriptors(
            {
                "*A": ["*B"],
                "*C": ["*D"],
            }
        )

        # Build traditionally
        builder = Builder(eads)
        relation = builder.build_traditional(descriptors)

        return relation

    @pytest.fixture
    def test_reaction(self):
        # Define dummy Species
        species_A = Species("A", -1, True)
        species_H2O = Species("H2O", -8, False)
        species_B = Species("B", -10, True)

        # Setup a test Reaction
        reaction = Reaction(
            [
                ReactionStep(
                    reactants={species_A: 1, species_H2O: 1},
                    products={species_B: 1},
                ),
            ]
        )

        return reaction

    def test_convert_step(self, eads_relation, test_reaction):
        converter = AdsorbToDeltaE(eads_relation, test_reaction)

        step_array = converter._convert_step(test_reaction[0])

        assert isinstance(step_array, np.ndarray)
        assert np.allclose(step_array, np.array([9.0, 0.0, -1.0]))

    def test_convert(self, eads_relation, test_reaction):
        converter = AdsorbToDeltaE(eads_relation, test_reaction)

        deltaE_relation = converter.convert()

        assert isinstance(deltaE_relation, DeltaERelation)

        assert len(deltaE_relation.coefficients) == len(test_reaction)
