"""Handle adsorption (free) energies for linear scaling relations."""


import pandas as pd


class Eads:
    """Handle adsorption energies as pandas.DataFrame.

    Expect data as a pd.DataFrame in the following format:
                   *CO2   *COOH  ...   *OCH3  *O     *OH
        Cu@g-C3N4  0.89   4.37   ...   3.98  -1.73   0.17
        Ni@C2N    -4.57  -4.95   ...  -0.93  -2.81  -3.21
        ......
        Pt@SiO2   -2.36   3.69   ...   3.12   0.29   4.84
        Au@Al2O3   2.15  -2.35   ...   1.36   1.07   4.56

    where:
        - Column headers (0th row) should be adsorbate names.
        - Row headers (0th column) should be sample names.

    Attributes:
        df (pd.DataFrame): The DataFrame containing adsorption energy data.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize the Eads class with a DataFrame."""
        # Take and check arg: df
        self.df = df
        self._check_df()

    def _check_df(self) -> bool:
        """Check the validity of the self.df.

        Ensures that the DataFrame is of type pd.DataFrame and attempts to
            convert its elements to float type.

        Returns:
            bool: True if DataFrame is valid, False otherwise.

        Raises:
            TypeError: If the input data is not of type pd.DataFrame.
            ValueError: If conversion of DataFrame elements to float fails.
        """
        if isinstance(self.df, pd.DataFrame):
            try:
                self.df = self.df.astype(float)
            except ValueError as e:
                raise ValueError(f"Please double-check input data: {e}.")

        else:
            raise TypeError("Expect data as pd.DataFrame type.")

        return True

    def add_adsorbate(self, name: str, energies: list[float]) -> None:
        """Append a new adsorbate column.

        Args:
            name (str): The name of the new adsorbate column.
            energies (list[float]): List of energies corresponding to the
                new adsorbate.

        Raises:
            ValueError: If the length of the new adsorbate energies doesn't
                match the number of samples, or if the adsorbate exists.
        """
        # Check new entry length
        if len(energies) != len(self.df):
            raise ValueError(
                "New adsorbate energies length doesn't match others."
            )

        if name in self.df.columns.values:
            raise ValueError(f"Adsorbate {name} already exists.")

        else:
            self.df[name] = energies

    def add_sample(self, name: str, energies: list[float]) -> None:
        """Append a new sample row.

        Args:
            name (str): The name of the new sample row.
            energies (list[float]): List of energies corresponding to
                the new sample.

        Raises:
            ValueError: If the length of the new sample energies doesn't match
                the number of adsorbates, or if the sample name already exists.
        """
        if len(energies) != len(self.df.columns):
            raise ValueError(
                "New sample energies length doesn't match others."
            )

        if name in self.df.index:
            raise ValueError(f"Sample {name} already exists.")

        else:
            self.df.loc[name] = energies

    def remove_adsorbate(self, name: str) -> None:
        """Remove an adsorbate (column).

        Args:
            name (str): The name of the adsorbate column to be removed.
        """
        self.df.drop(columns=name, inplace=True)

    def remove_sample(self, name: str) -> None:
        """Remove a sample (row).

        Args:
            name (str): The name of the sample row to be removed.
        """
        self.df.drop(index=name, inplace=True)

    def get_adsorbates(self) -> list[str]:
        """Get adsorbate names from column headers."""
        return self.df.columns.values.tolist()

    def get_samples(self) -> list[str]:
        """Get sample names from row headers."""
        return self.df.index.tolist()

    def sort_df(self, targets: list[str] = ["column", "row"]) -> None:
        """Sort columns/rows of df."""
        if not set(targets) <= {"column", "row"}:
            raise ValueError(
                "Invalid target values. Should be 'column', 'row', or both."
            )

        if "column" in targets:
            self.df = self.df.sort_index(axis=1)

        if "row" in targets:
            self.df = self.df.sort_index()

    def set_groups(self, grouping: list[list[str]]) -> None:
        """Group adsorbate by their names."""
        # TODO
        pass
