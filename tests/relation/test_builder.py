from math import isclose

import numpy as np
import pandas as pd
import pytest

from scaling.data.eads import Eads
from scaling.relation import Builder
from scaling.utils import PROJECT_ROOT


class Test_builder:
    test_data_csv = PROJECT_ROOT / "tests" / "relation" / "relation_data.csv"

    @pytest.fixture(autouse=True)
    def setup_data_load(self):
        # Data
        test_df = pd.read_csv(self.test_data_csv, index_col=[0], header=[0])
        self.eads = Eads(test_df)

    def test_properties(self):
        # Property: descriptors
        descriptors = ["*CO", "*OH"]

        # Property: ratios
        ratios = {
            "*COOH": [0.2, 0.8],
            "*O": [0.2, 0.8],
        }

        # Property: groups
        groups = {
            "*CO": ["*CO2", "*COOH"],
            "*OH": ["*O", "OCH3"],
        }

        # Property: method
        method = "traditional"

        Builder(
            self.eads,
            descriptors,
            ratios,
            groups,
            method,
        )

    def test_invalid_properties(self):
        # TODO
        pass

    def test_build_composite_descriptor(self):
        # Prepare descriptors
        descriptors = ["*A", "*D"]

        builder = Builder(
            self.eads,
            descriptors,
        )

        # Sum of *A and *D should be zeros
        comp_des_0 = builder._build_composite_descriptor(
            names=["*A", "*D"], ratios=[0.5, 0.5]
        )

        assert np.allclose(comp_des_0, np.zeros(6))

        # Should be just *D
        comp_des_1 = builder._build_composite_descriptor(
            names=["*A", "*D"], ratios=[0, 1]
        )

        assert np.allclose(
            comp_des_1, np.array([0, -0.1, -0.2, -0.3, -0.4, -0.5])
        )

        # Should be just *A
        comp_des_2 = builder._build_composite_descriptor(
            names=["*A", "*D"], ratios=[1, 0]
        )

        assert np.allclose(comp_des_2, np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5]))

    def test_builder(self):
        # Prepare descriptors
        descriptors = [
            "*A",
        ]
        groups = {
            "*A": [
                "*B",
            ],
        }

        builder = Builder(self.eads, groups=groups)

        # Test fit *B with *A
        relation = builder._builder(
            descriptors,
            ratios=[
                1.0,
            ],
        )

        # Check regression results
        assert relation.dim == 1
        assert isclose(relation.metrics["*B"], 1, abs_tol=0.01)
        assert np.allclose(relation.coefficients["*B"], [10, 0])
