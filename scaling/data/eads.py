"""Handle adsorption (free) energies for linear scaling relations."""


import pandas as pd


class Eads:
    """Handle adsorption energies for linear scaling relations.

    Expect data as a pd.DataFrame in the following format:
                   *CO2   *COOH  ...  *OCH3  *O     *OH
        Cu@g-C3N4  0.89   4.37   ...  3.98  -1.73   0.17
        Ni@C2N    -4.57  -4.95  ...  -0.93  -2.81  -3.21
        ...
        Pt@SiO2   -2.36   3.69   ...  3.12   0.29   4.84
        Au@Al2O3   2.15  -2.35  ...   1.36   1.07   4.56

    Note:
        - Column headers (0th row) should be adsorbate names.
        - Row headers (0th column) should be sample names.

    Attributes:
        df (pd.DataFrame): The DataFrame containing adsorption energy data.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize the Eads class with a DataFrame."""
        # Check and take arg: df
        if isinstance(df, pd.DataFrame):
            try:
                self.df = df.astype(float)
            except ValueError as e:
                raise ValueError(f"Please double-check input data: {e}.")

        else:
            raise TypeError("Expect data as pd.DataFrame type.")

    def get_adsorbates(self) -> list[str]:
        """Get adsorbate names from column headers."""
        return self.df.columns.values.tolist()

    def get_samples(self) -> list[str]:
        """Get sample names from row headers."""
        return self.df.index.tolist()
