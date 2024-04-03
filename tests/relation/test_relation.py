import numpy as np
import pytest

from cat_scaling.relation.relation import DeltaERelation, EadsRelation


class Test_EadsRelation:
    test_metrics = {"a": 0.8, "b": 0.9}
    test_ratios = {"b": {"a": 1.0}}

    def test_init(self):
        test_coef = {
            "a": [0.1, 0.2, 0.3],
            "b": [0.0, 0.1, 0.2],
        }

        test_intercepts = {"a": 0, "b": 1}

        relation = EadsRelation(
            test_coef, test_intercepts, self.test_metrics, self.test_ratios
        )

        assert relation.dim == len(test_coef["a"])

    def test_invalid_properties(self):
        test_coef = {
            "a": [0.1, 0.2, 0.3],
            "b": [0.0, 0.1, 0.2],
        }

        # Test invalid intercepts
        with pytest.raises(TypeError, match="intercepts should be a dict"):
            test_intercepts = [1, 0]

            EadsRelation(
                test_coef, test_intercepts, self.test_metrics, self.test_ratios
            )

        with pytest.raises(TypeError, match="intercept value should be float"):
            test_intercepts = {"a": "0", "b": 1}

            EadsRelation(
                test_coef, test_intercepts, self.test_metrics, self.test_ratios
            )

        # Test invalid metrics
        test_intercepts = {"a": 0, "b": 1}

        with pytest.raises(TypeError, match="metric should be a dict"):
            invalid_metrics = [1, 0]
            EadsRelation(
                test_coef, test_intercepts, invalid_metrics, self.test_ratios
            )

        with pytest.raises(TypeError, match="metric values should be float"):
            invalid_metrics = {"a": "0.8", "b": 0.9}
            EadsRelation(
                test_coef, test_intercepts, invalid_metrics, self.test_ratios
            )

        # Test warning for low metrics
        with pytest.warns(UserWarning, match="Low metrics for"):
            warn_metrics = {"a": 0.1, "b": 0.9}
            EadsRelation(
                test_coef, test_intercepts, warn_metrics, self.test_ratios
            )

        # Test invalid ratios
        with pytest.raises(ValueError, match="Each ratio_dict must be a dict"):
            invalid_ratios = {"b": [0.1, 0.2]}
            EadsRelation(
                test_coef, test_intercepts, self.test_metrics, invalid_ratios
            )

        with pytest.raises(
            ValueError, match="Ratios for each species should sum to one"
        ):
            invalid_ratios = {"b": {"a": 0.9}}
            EadsRelation(
                test_coef, test_intercepts, self.test_metrics, invalid_ratios
            )

        with pytest.raises(
            ValueError, match="Ratio dict must have the same length"
        ):
            invalid_ratios = {"b": {"a": 1.0}, "c": {"d": 0.9, "e": 0.1}}
            EadsRelation(
                test_coef, test_intercepts, self.test_metrics, invalid_ratios
            )

    def test_str(self):
        test_coef = {
            "a": [0.1, 0.2, 0.3],
            "b": [0.0, 0.1, 0.2],
        }

        test_intercept = {"a": 0, "b": 1}
        test_metrics = {"a": 0.8, "b": 0.9}
        test_ratios = {"b": {"a": 1.0}}

        relation = EadsRelation(
            test_coef, test_intercept, test_metrics, test_ratios
        )

        assert "coef_0     coef_1     coef_2     intercept" in str(relation)
        assert "Adsorbate Metrics" in str(relation)
        assert "Adsorbate Ratios" in str(relation)

    def test_invalid_data_dtype(
        self,
    ):
        invalid_coef = []
        test_intercept = {"a": 0, "b": 1}
        with pytest.raises(
            TypeError, match="Coefficients must be a dictionary"
        ):
            EadsRelation(
                invalid_coef,
                test_intercept,
                self.test_metrics,
                self.test_ratios,
            )

    def test_invalid_species_name(self):
        invalid_coef = {
            1: [1, 2, 3],
            "b": [0.1, 0.2, 0.3],
        }
        test_intercept = {"a": 0, "b": 1}
        with pytest.raises(TypeError, match="Species name must be strings"):
            EadsRelation(
                invalid_coef,
                test_intercept,
                self.test_metrics,
                self.test_ratios,
            )

    def test_invalid_coef_type(self):
        invalid_coef_0 = {
            "a": "1",
            "b": [0.1, 0.2, 0.3],
        }
        test_intercept = {"a": 0, "b": 1}
        with pytest.raises(
            TypeError, match="Input coefficients must be lists"
        ):
            EadsRelation(
                invalid_coef_0,
                test_intercept,
                self.test_metrics,
                self.test_ratios,
            )

        invalid_coef_1 = {
            "a": ["1", "2"],
            "b": [0.1, 0.2, 0.3],
        }
        test_intercept = {"a": 0, "b": 1}
        with pytest.raises(TypeError, match="Coefficients must be floats"):
            EadsRelation(
                invalid_coef_1,
                test_intercept,
                self.test_metrics,
                self.test_ratios,
            )

    def test_coef_length_mismatch(self):
        invalid_coef = {
            "a": [0.1, 0.2],
            "b": [0.1, 0.2, 0.3],
        }
        test_intercept = {"a": 0, "b": 1}
        with pytest.raises(
            ValueError, match="All coefficients must have the same length"
        ):
            EadsRelation(
                invalid_coef,
                test_intercept,
                self.test_metrics,
                self.test_ratios,
            )


class Test_DeltaERelation:
    def test_init(self):
        delta_E_relation = DeltaERelation(
            coefficients=[np.random.rand(3), np.random.rand(3)]
        )
        assert isinstance(delta_E_relation, DeltaERelation)

        assert delta_E_relation.dim == 2

    def test_invalid_properties(self):
        # Test invalid coefficients
        with pytest.raises(TypeError, match="Coefficients should be a list"):
            DeltaERelation(coefficients=np.random.rand(3))

        with pytest.raises(
            TypeError, match="All coefficients should be numpy arrays"
        ):
            DeltaERelation(coefficients=[np.random.rand(3), [0, 1, 2]])

        with pytest.raises(
            ValueError,
            match="All coefficient arrays should have the same length",
        ):
            DeltaERelation(coefficients=[np.random.rand(3), np.random.rand(4)])

    def test_eval_limit_potential_2D(self):
        pass  # TODO
