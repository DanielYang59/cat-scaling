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

    def test_add_sample(self, setup_class):
        self.eads.add_sample("new_sample", list(range(6)))
        assert "new_sample" in self.eads.samples

    def test_remove_adsorbate(self, setup_class):
        self.eads.remove_adsorbate("*CO2")
        assert "*CO2" not in self.eads.adsorbates

    def test_remove_sample(self, setup_class):
        self.eads.remove_sample("Cu@g-C3N4")
        assert "Cu@g-C3N4" not in self.eads.samples

    def test_sort_df(self, setup_class):
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
