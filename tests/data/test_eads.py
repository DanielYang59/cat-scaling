"""Pytest unit test for class Eads."""

# TODO: use a single setup method

import numpy as np
import pandas as pd

from scaling.data.eads import Eads
from scaling.utils import PROJECT_ROOT


class Test_eads:
    test_data_csv = PROJECT_ROOT / "tests" / "data" / "example_eads_data.csv"

    def test_init(self):
        test_df = pd.read_csv(self.test_data_csv, index_col=[0], header=[0])
        eads = Eads(test_df)

        assert isinstance(eads.data, pd.DataFrame)

    def test_property_adsorbates_samples(self):
        # Setup
        test_df = pd.read_csv(self.test_data_csv, index_col=[0], header=[0])
        eads = Eads(test_df)

        # Test property: adsorbates
        adsorbates = eads.adsorabtes

        assert adsorbates == ["*CO2", "*COOH", "*CO", "*OCH3", "*O", "*OH"]

        # Test property: samples
        samples = eads.samples

        assert samples == [
            "Cu@g-C3N4",
            "Ni@C2N",
            "Fe@BN",
            "Co@BN",
            "Pt@SiO2",
            "Au@Al2O3",
        ]

    def test_get_adsorbate_sample(self):
        # Setup
        test_df = pd.read_csv(self.test_data_csv, index_col=[0], header=[0])
        eads = Eads(test_df)

        # Test method: get_adsorbate
        col = eads.get_adsorbate("*CO")

        assert np.array_equal(
            col, np.array([3.64, -1.45, 1.51, 0.52, 1.74, -3.97])
        )

        # Test method: get_sample
        row = eads.get_sample("Cu@g-C3N4")

        assert np.array_equal(
            row, np.array([0.89, 4.37, 3.64, 3.98, -1.73, 0.17])
        )

    def test_add_adsorbate(self):
        test_df = pd.read_csv(self.test_data_csv, index_col=[0], header=[0])
        eads = Eads(test_df)

        eads.add_adsorbate("new_adsorbate", list(range(6)))
        assert "new_adsorbate" in eads.adsorabtes

    def test_add_sample(self):
        test_df = pd.read_csv(self.test_data_csv, index_col=[0], header=[0])
        eads = Eads(test_df)

        eads.add_sample("new_sample", list(range(6)))
        assert "new_sample" in eads.samples

    def test_remove_adsorbate(self):
        test_df = pd.read_csv(self.test_data_csv, index_col=[0], header=[0])
        eads = Eads(test_df)

        eads.remove_adsorbate("*CO2")
        assert "*CO2" not in eads.adsorabtes

    def test_remove_sample(self):
        test_df = pd.read_csv(self.test_data_csv, index_col=[0], header=[0])
        eads = Eads(test_df)

        eads.remove_sample("Cu@g-C3N4")
        assert "Cu@g-C3N4" not in eads.samples

    def test_sort_df(self):
        test_df = pd.read_csv(self.test_data_csv, index_col=[0], header=[0])
        eads = Eads(test_df)
        eads.sort_data(targets=["column", "row"])

        assert eads.adsorabtes == [
            "*CO",
            "*CO2",
            "*COOH",
            "*O",
            "*OCH3",
            "*OH",
        ]

        assert eads.samples == [
            "Au@Al2O3",
            "Co@BN",
            "Cu@g-C3N4",
            "Fe@BN",
            "Ni@C2N",
            "Pt@SiO2",
        ]
