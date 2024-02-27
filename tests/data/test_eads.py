"""Pytest unit test for class Eads."""


import pandas as pd

from scaling.data.eads import Eads
from scaling.utils import PROJECT_ROOT


class Test_eads:
    test_data_csv = PROJECT_ROOT / "tests" / "data" / "example_eads_data.csv"

    def test_init(self):
        test_df = pd.read_csv(self.test_data_csv, index_col=[0], header=[0])
        eads = Eads(test_df)

        assert isinstance(eads.df, pd.DataFrame)

    def test_get_adsorbates(self):
        test_df = pd.read_csv(self.test_data_csv, index_col=[0], header=[0])
        eads = Eads(test_df)

        adsorbates = eads.get_adsorbates()

        assert adsorbates == ["*CO2", "*COOH", "*CO", "*OCH3", "*O", "*OH"]

    def test_get_samples(self):
        test_df = pd.read_csv(self.test_data_csv, index_col=[0], header=[0])
        eads = Eads(test_df)

        samples = eads.get_samples()

        assert samples == [
            "Cu@g-C3N4", "Ni@C2N", "Fe@BN", "Co@BN", "Pt@SiO2", "Au@Al2O3"
        ]

    def test_add_adsorbate(self):
        test_df = pd.read_csv(self.test_data_csv, index_col=[0], header=[0])
        eads = Eads(test_df)

        eads.add_adsorbate("new_adsorbate", list(range(6)))
        assert "new_adsorbate" in eads.get_adsorbates()

    def test_add_sample(self):
        test_df = pd.read_csv(self.test_data_csv, index_col=[0], header=[0])
        eads = Eads(test_df)

        eads.add_sample("new_sample", list(range(6)))
        assert "new_sample" in eads.get_samples()

    def test_sort_df(self):
        test_df = pd.read_csv(self.test_data_csv, index_col=[0], header=[0])
        eads = Eads(test_df)
        eads.sort_df(targets=["column", "row"])

        assert eads.get_adsorbates() == [
            "*CO", "*CO2", "*COOH", "*O", "*OCH3", "*OH"
        ]

        assert eads.get_samples() == [
            "Au@Al2O3", "Co@BN", "Cu@g-C3N4", "Fe@BN", "Ni@C2N", "Pt@SiO2"
        ]
