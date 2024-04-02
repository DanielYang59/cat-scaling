import copy

import numpy as np
import pandas as pd
import pytest

from cat_scaling.data.eads import Eads
from cat_scaling.utils import PROJECT_ROOT


class Test_eads:
    test_data_csv = PROJECT_ROOT / "tests" / "data" / "example_eads_data.csv"

    @classmethod
    @pytest.fixture()
    def setup_class(cls):
        cls.test_df = pd.read_csv(cls.test_data_csv, index_col=[0], header=[0])
        cls.eads = Eads(cls.test_df)

    def test_init(self, setup_class):
        assert isinstance(
            self.eads.data,
            pd.DataFrame,
        )

    @pytest.mark.filterwarnings("ignore:Setting an item of incompatible dtype")
    def test_invalid_dtype(self, setup_class):
        """Eads expect data with dtype as float."""
        # Inject an invalid data type
        invalid_data = copy.deepcopy(self.eads.data)
        invalid_data.iloc[0, 0] = "wrong_data_type"  # should be float

        with pytest.raises(ValueError):
            Eads(data=invalid_data)

    def test_invalid_dataframe_type(self):
        """Eads expect data as pd.DataFrame."""
        with pytest.raises(TypeError):
            Eads(data=np.array([0, 1, 2]))

    def test_property_adsorbates_samples(self, setup_class):
        # Test property: adsorbates
        adsorbates = self.eads.adsorbates

        assert adsorbates == ["*CO2", "*COOH", "*CO", "*OCH3", "*O", "*OH"]

        # Test property: samples
        samples = self.eads.samples

        assert samples == [
            "Cu@g-C3N4",
            "Ni@C2N",
            "Fe@BN",
            "Co@BN",
            "Pt@SiO2",
            "Au@Al2O3",
        ]

    def test_from_csv(self):
        """Test initialize Eads from csv file."""
        eads = Eads.from_csv(self.test_data_csv)
        assert isinstance(eads, Eads)

    def test_from_wrong_format(self):
        """Eads expect a csv file for from_csv method."""
        with pytest.raises(ValueError):
            Eads.from_csv(self.test_data_csv.with_suffix(".null"))

    def test_get_adsorbate_sample(self, setup_class):
        # Test method: get_adsorbate
        col = self.eads.get_adsorbate("*CO")

        assert np.array_equal(
            col, np.array([3.64, -1.45, 1.51, 0.52, 1.74, -3.97])
        )

        # Test method: get_sample
        row = self.eads.get_sample("Cu@g-C3N4")

        assert np.array_equal(
            row, np.array([0.89, 4.37, 3.64, 3.98, -1.73, 0.17])
        )

    def test_add_adsorbate(self, setup_class):
        self.eads.add_adsorbate("new_adsorbate", list(range(6)))
        assert "new_adsorbate" in self.eads.adsorbates

    def test_add_adsorbate_wrong_length(self, setup_class):
        """Test add a adsorbate column but with inconsistent length."""
        with pytest.raises(ValueError):
            self.eads.add_adsorbate("new_adsorbate", list(range(10)))

    def test_add_existing_adsorbate(self, setup_class):
        with pytest.raises(ValueError):
            self.eads.add_adsorbate("*CO2", list(range(10)))

    def test_add_sample(self, setup_class):
        self.eads.add_sample("new_sample", list(range(6)))
        assert "new_sample" in self.eads.samples

    def test_add_sample_wrong_length(self, setup_class):
        """Test add a adsorbate column but with inconsistent length."""
        with pytest.raises(ValueError):
            self.eads.add_sample("new_sample", list(range(10)))

    def test_add_existing_sample(self, setup_class):
        with pytest.raises(ValueError):
            self.eads.add_sample("Cu@g-C3N4", list(range(6)))

    def test_remove_adsorbate(self, setup_class):
        self.eads.remove_adsorbate("*CO2")
        assert "*CO2" not in self.eads.adsorbates

    def test_remove_sample(self, setup_class):
        self.eads.remove_sample("Cu@g-C3N4")
        assert "Cu@g-C3N4" not in self.eads.samples

    def test_sort_date(self, setup_class):
        self.eads.sort_data(targets={"column", "row"})

        assert self.eads.adsorbates == [
            "*CO",
            "*CO2",
            "*COOH",
            "*O",
            "*OCH3",
            "*OH",
        ]

        assert self.eads.samples == [
            "Au@Al2O3",
            "Co@BN",
            "Cu@g-C3N4",
            "Fe@BN",
            "Ni@C2N",
            "Pt@SiO2",
        ]

    def test_sort_date_invalid_targets(self, setup_class):
        with pytest.raises(ValueError):
            self.eads.sort_data(targets={"invalid", "row"})

        with pytest.raises(ValueError):
            self.eads.sort_data(targets={"column", "row", "invalid"})
