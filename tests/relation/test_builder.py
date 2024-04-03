from math import isclose

import numpy as np
import pandas as pd
import pytest

from cat_scaling.data.eads import Eads
from cat_scaling.relation import Builder, Descriptors
from cat_scaling.utils import PROJECT_ROOT


class Test_builder:
    test_data_csv = PROJECT_ROOT / "tests" / "relation" / "relation_data.csv"

    @pytest.fixture(autouse=True)
    def setup_data_load(self):
        # Data
        test_df = pd.read_csv(self.test_data_csv, index_col=[0], header=[0])
        self.eads = Eads(test_df)

    def test_invalid_data(self):
        with pytest.raises(TypeError, match="Expect data as 'Eads' type"):
            Builder(data="data")

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

        # Test invalid ratio sum (should sum to 1.0)
        with pytest.raises(ValueError, match="Ratios should sum to 1.0"):
            builder._build_composite_descriptor({"*A": 0.5, "*D": 0})

    def test_builder(self):
        # Prepare Builder
        builder = Builder(self.eads)

        # Test fitting *B with *A
        coefs, intercept, metrics = builder._builder(
            adsorbate_name="*B", ratios={"*A": 1}
        )

        # Check regression results
        assert len(coefs) == 1
        assert np.allclose(coefs, [10], atol=0.01)
        assert isclose(intercept, 0, abs_tol=0.01)
        assert isclose(metrics, 1, abs_tol=0.01)

    def test_build_traditional(self):
        """Test build with traditional single descriptor method.
        Descriptor A: [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        Descriptor B: [0, 1, 2, 3, 4, 5]

        Thus B = 10 * A + 0, coefficient being 10 and intercept being 0

        """
        # Prepare Builder
        builder = Builder(self.eads)

        # Define descriptors
        descriptors = Descriptors(groups={"*A": ["*B"]})

        # Test build *B with descriptor *A
        relation = builder.build_traditional(descriptors)

        # Check scaling results
        assert relation.dim == 1  # only one descriptor *B
        assert np.allclose(relation.coefficients["*B"], [10], atol=0.01)
        assert isclose(relation.intercepts["*B"], 0, abs_tol=0.01)
        assert isclose(relation.metrics["*B"], 1.0, abs_tol=0.01)

        # Check descriptor ratio
        assert isclose(relation.ratios["*B"]["*A"], 1.0, abs_tol=0.01)

    def test_build_traditional_invalid(self):
        with pytest.raises(
            ValueError,
            match="Group member for traditional builder cannot be None",
        ):
            descriptors = Descriptors(groups={"*A": None})

            builder = Builder(self.eads)
            builder.build_traditional(descriptors)

    def test_build_adaptive(self):
        """Test build with adaptive descriptor method.

        Descriptor A: [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        Descriptor D: [0, -0.1, -0.2, -0.3, -0.4, -0.5]

        As such *A and *D are strongly correlated, it's thus
        expected a composite descriptor with (*A * 1 + *D * 0)
        to be the most suitable descriptor.
        """
        # Define descriptors
        descriptors = Descriptors(
            {
                "*A": None,
                "*D": None,
            }
        )

        # Prepare Builder
        builder = Builder(self.eads)

        # Test with adaptive descriptors *A and *D
        relation = builder.build_adaptive(descriptors, step_length=1)

        # Check scaling results
        assert relation.dim == 2

        assert np.allclose(relation.coefficients["*A"], [0, -1], atol=0.01)
        assert isclose(relation.intercepts["*A"], 0, abs_tol=0.01)
        assert isclose(relation.metrics["*A"], 1.0, abs_tol=0.01)

        assert isclose(relation.ratios["*B"]["*D"], 1, abs_tol=0.01)

    def test_build_adaptive_invalid_warn(self):
        # Test invalid step length
        builder = Builder(self.eads)
        descriptors = Descriptors(
            {
                "*A": None,
                "*D": None,
            }
        )

        with pytest.raises(ValueError, match="Illegal step length"):
            builder.build_adaptive(descriptors, step_length="10")

        with pytest.raises(ValueError, match="Illegal step length"):
            builder.build_adaptive(descriptors, step_length=200)

        # Test warning for too small/large step length
        with pytest.warns(
            UserWarning, match="Large step length may harm accuracy"
        ):
            builder.build_adaptive(descriptors, step_length=20)

        with pytest.warns(
            UserWarning, match="Small step length may slow down searching"
        ):
            # NOTE: warning: this may slow down unit test
            builder.build_adaptive(descriptors, step_length=0.09)

        # Test invalid number of descriptors
        with pytest.raises(
            ValueError, match="Expect two descriptors for adaptive method"
        ):
            descriptors = Descriptors(
                {
                    "*A": None,
                }
            )
            builder.build_adaptive(descriptors, step_length=1)
