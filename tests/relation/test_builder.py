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
        # TODO: need update
        pass

    def test_invalid_properties(self):
        # TODO
        pass

    def test_build_composite_descriptor(self):
        builder = Builder(self.eads)

        # Sum of *A and *D should be zeros
        ratios = {"*A": 0.5, "*D": 0.5}
        comp_des_0 = builder._build_composite_descriptor(ratios)

        assert np.allclose(comp_des_0, np.zeros(6))

        # Should be just *D
        ratios = {"*A": 0, "*D": 1}
        comp_des_1 = builder._build_composite_descriptor(ratios)

        assert np.allclose(
            comp_des_1, np.array([0, -0.1, -0.2, -0.3, -0.4, -0.5])
        )

        # Should be just *A
        ratios = {"*A": 1, "*D": 0}
        comp_des_2 = builder._build_composite_descriptor(ratios)

        assert np.allclose(comp_des_2, np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5]))

    def test_builder(self):
        # Prepare Builder
        builder = Builder(self.eads)

        # Test fitting *B with *A
        coefs, intercept, metrics = builder._builder(
            spec_name="*B", ratios={"*A": 1}
        )

        # Check regression results
        assert len(coefs) == 1
        assert np.allclose(coefs, [10], atol=0.01)
        assert isclose(intercept, 0, abs_tol=0.01)
        assert isclose(metrics, 1, abs_tol=0.01)

    def test_build_traditional(self):
        # Prepare Builder
        builder = Builder(self.eads)

        # Test build *B with descriptor *A
        relation = builder.build_traditional(groups={"*A": ["*B"]})

        # Check regression results
        assert relation.dim == 1
        assert np.allclose(relation.coefficients["*B"], [10], atol=0.01)
        assert isclose(relation.intercepts["*B"], 0, abs_tol=0.01)
        assert isclose(relation.metrics["*B"], 1.0, abs_tol=0.01)

        # Check descriptor ratio
        assert isclose(relation.ratios["*B"]["*A"], 1.0, abs_tol=0.01)
