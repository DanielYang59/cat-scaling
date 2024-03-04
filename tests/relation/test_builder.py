import pandas as pd

from scaling.data.eads import Eads
from scaling.relation import Builder
from scaling.utils import PROJECT_ROOT


class Test_builder:
    test_data_csv = PROJECT_ROOT / "tests" / "data" / "example_eads_data.csv"

    def test_properties(self):
        # Data
        test_df = pd.read_csv(self.test_data_csv, index_col=[0], header=[0])
        eads = Eads(test_df)

        # Property: descriptors
        descriptors = ["*CO", "*OH"]

        # Property: ratios
        ratios = {
            "*CO": [0.2, 0.8],
            "*OH": [0.2, 0.8],
        }

        # Property: groups
        groups = {
            "*CO": ["*CO2", "*COOH"],
            "*OH": ["*O", "OCH3"],
        }

        # Property: method
        method = "traditional"

        Builder(
            eads,
            descriptors,
            ratios,
            groups,
            method,
        )

    def test_build_composite_descriptor(self):
        pass

    def test_builder(self):
        pass
