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
