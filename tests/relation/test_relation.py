# TODO: avoid duplication in test setup for Test_EadsRelation


import pytest

from scaling.relation.relation import EadsRelation


class Test_EadsRelation:
    test_metrics = {"a": 0.8, "b": 0.9}
    test_ratios = {"b": {"a": 1.0}}

    def test_init(self):
        test_coef = {
            "a": [0.1, 0.2, 0.3],
            "b": [0.0, 0.1, 0.2],
        }

        test_intercept = {"a": 0, "b": 1}

        relation = EadsRelation(
            test_coef, test_intercept, self.test_metrics, self.test_ratios
        )

        assert relation.dim == len(test_coef["a"])

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
        with pytest.raises(TypeError) as excinfo:
            EadsRelation(
                invalid_coef,
                test_intercept,
                self.test_metrics,
                self.test_ratios,
            )

        assert str(excinfo.value) == "Coefficients must be a dictionary"

    def test_invalid_species_name(self):
        invalid_coef = {
            1: [1, 2, 3],
            "b": [0.1, 0.2, 0.3],
        }
        test_intercept = {"a": 0, "b": 1}
        with pytest.raises(TypeError) as excinfo:
            EadsRelation(
                invalid_coef,
                test_intercept,
                self.test_metrics,
                self.test_ratios,
            )

        assert str(excinfo.value) == "Species name must be strings"

    def test_invalid_coef_type(self):
        invalid_coef_0 = {
            "a": "1",
            "b": [0.1, 0.2, 0.3],
        }
        test_intercept = {"a": 0, "b": 1}
        with pytest.raises(TypeError) as excinfo:
            EadsRelation(
                invalid_coef_0,
                test_intercept,
                self.test_metrics,
                self.test_ratios,
            )

        assert str(excinfo.value) == "Input coefficients must be lists"

        invalid_coef_1 = {
            "a": ["1", "2"],
            "b": [0.1, 0.2, 0.3],
        }
        test_intercept = {"a": 0, "b": 1}
        with pytest.raises(TypeError) as excinfo:
            EadsRelation(
                invalid_coef_1,
                test_intercept,
                self.test_metrics,
                self.test_ratios,
            )

        assert str(excinfo.value) == "Coefficients must be floats"

    def test_coef_length_mismatch(self):
        invalid_coef = {
            "a": [0.1, 0.2],
            "b": [0.1, 0.2, 0.3],
        }
        test_intercept = {"a": 0, "b": 1}
        with pytest.raises(ValueError) as excinfo:
            EadsRelation(
                invalid_coef,
                test_intercept,
                self.test_metrics,
                self.test_ratios,
            )

        assert (
            str(excinfo.value) == "All coefficients must have the same length"
        )
