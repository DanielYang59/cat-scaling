import pytest

from scaling.relation import Relation


class Test_relation:
    def test_init(self):
        test_coef = {
            "a": [0.1, 0.2, 0.3],
            "b": [0.0, 0.1, 0.2],
        }

        test_intercept = {"a": 0, "b": 1}

        relation = Relation(test_coef, test_intercept)

        assert relation.dim == len(test_coef["a"])

    def test_str(self):
        test_coef = {
            "a": [0.1, 0.2, 0.3],
            "b": [0.0, 0.1, 0.2],
        }

        test_intercept = {"a": 0, "b": 1}

        test_metrics = {"a": 0.8, "b": 0.9}

        test_ratios = {"b": {"a": 1.0}}

        relation = Relation(
            test_coef, test_intercept, test_metrics, test_ratios
        )

        assert str(relation)

    def test_invalid_data_dtype(
        self,
    ):
        invalid_coef = []
        test_intercept = {"a": 0, "b": 1}
        with pytest.raises(TypeError) as excinfo:
            Relation(invalid_coef, test_intercept)

        assert str(excinfo.value) == "Coefficients must be a dictionary"

    def test_invalid_species_name(self):
        invalid_coef = {
            1: [1, 2, 3],
            "b": [0.1, 0.2, 0.3],
        }
        test_intercept = {"a": 0, "b": 1}
        with pytest.raises(TypeError) as excinfo:
            Relation(invalid_coef, test_intercept)

        assert str(excinfo.value) == "Species name must be strings"

    def test_invalid_coef_type(self):
        invalid_coef = {
            "a": "1",
            "b": [0.1, 0.2, 0.3],
        }
        test_intercept = {"a": 0, "b": 1}
        with pytest.raises(TypeError) as excinfo:
            Relation(invalid_coef, test_intercept)

        assert str(excinfo.value) == "Input coefficients must be lists"

    def test_invalid_coef_dtype(
        self,
    ):
        invalid_coef = {
            "a": ["1", "2"],
            "b": [0.1, 0.2, 0.3],
        }
        test_intercept = {"a": 0, "b": 1}
        with pytest.raises(TypeError) as excinfo:
            Relation(invalid_coef, test_intercept)

        assert str(excinfo.value) == "Coefficients must be floats"

    def test_coef_length_mismatch(self):
        invalid_coef = {
            "a": [0.1, 0.2],
            "b": [0.1, 0.2, 0.3],
        }
        test_intercept = {"a": 0, "b": 1}
        with pytest.raises(ValueError) as excinfo:
            Relation(invalid_coef, test_intercept)

        assert (
            str(excinfo.value) == "All coefficients must have the same length"
        )
