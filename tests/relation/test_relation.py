import pytest

from scaling.relation import Relation


class Test_relation:
    def test_init(self):
        test_coef = {
            "a": [0.1, 0.2, 0.3],
            "b": [0.0, 0.1, 0.2],
        }
        relation = Relation(test_coef)

        assert relation.dim == 2

    def test_invalid_data_dtype(self):
        invalid_coef = []
        with pytest.raises(TypeError) as excinfo:
            Relation(invalid_coef)

        assert str(excinfo.value) == "Coefficients must be a dictionary"

    def test_invalid_species_name(self):
        invalid_coef = {
            1: [1, 2, 3],
            "b": [0.1, 0.2, 0.3],
        }
        with pytest.raises(TypeError) as excinfo:
            Relation(invalid_coef)

        assert str(excinfo.value) == "Species name must be strings"

    def test_invalid_coef_type(self):
        invalid_coef = {
            "a": "1",
            "b": [0.1, 0.2, 0.3],
        }
        with pytest.raises(TypeError) as excinfo:
            Relation(invalid_coef)

        assert str(excinfo.value) == "Input coefficients must be lists"

    def test_invalid_coef_dtype(self):
        invalid_coef = {
            "a": ["1", "2"],
            "b": [0.1, 0.2, 0.3],
        }
        with pytest.raises(TypeError) as excinfo:
            Relation(invalid_coef)

        assert str(excinfo.value) == "Coefficients must be floats"

    def test_coef_length_mismatch(self):
        invalid_coef = {
            "a": [0.1, 0.2],
            "b": [0.1, 0.2, 0.3],
        }
        with pytest.raises(ValueError) as excinfo:
            Relation(invalid_coef)

        assert str(excinfo.value) == "All coefficients must have the same length"
