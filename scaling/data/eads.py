"""Handle adsorption (free) energies for linear scaling relations."""

from typing import Optional

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

    def __init__(
        self,
        data: pd.DataFrame,
        groups: Optional[dict[str, list[str]]] = None,
    ) -> None:
        """Initialize the Eads class with a DataFrame."""

        # Set property: data
        self.data = data
        self.groups = groups

    @property
    def data(self) -> pd.DataFrame:
        """Eads data as a pd.DataFrame.

        Expect data in the following format (as a pd.DataFrame):
                    *CO2   *COOH  ...   *OCH3  *O     *OH
            Cu@g-C3N4  0.89   4.37   ...   3.98  -1.73   0.17
            Ni@C2N    -4.57  -4.95   ...  -0.93  -2.81  -3.21
            ......
            Pt@SiO2   -2.36   3.69   ...   3.12   0.29   4.84
            Au@Al2O3   2.15  -2.35   ...   1.36   1.07   4.56

            where:
                - Column headers (0th row) should be adsorbate names.
                - Row headers (0th column) should be sample names.
        """

        return self._data

    @data.setter
    def data(self, data: pd.DataFrame):
        """Check and set self.data.

        Ensures that the DataFrame is of type pd.DataFrame and attempts to
            convert its elements to float type.

        Raises:
            TypeError: If the input data is not of type pd.DataFrame.
            ValueError: If conversion of DataFrame elements to float fails.
        """

        if isinstance(data, pd.DataFrame):
            try:
                data = data.astype(float)
            except ValueError as e:
                raise ValueError(f"Please double-check input data: {e}.")

        else:
            raise TypeError("Expect data as pd.DataFrame type.")

        self._data = data

    @property
    def adsorabtes(self) -> list[str]:
        """Adsorbate names (from column headers)."""

        return self.data.columns.values.tolist()

    @property
    def samples(self) -> list[str]:
        """Sample names (from row headers)."""

        return self.data.index.tolist()

    @property
    def groups(self):
        """Group adsorbates by specified names.

        Args:
            groups (dict[str, list[str]]): A dictionary defining groups where
                keys are group names and values are lists of
                adsorbate names belonging to each group.

        Raises:
            RuntimeError: If the adsorbates in the provided groups do not
                match the adsorbates in the DataFrame.

        Example:
            {
                "carbon": ["*CO2", "*CO"],
                "oxygen": ["*O", "*OH"],
                "hydrogen": ["*H"],
            }
        """
        return self._groups

    @groups.setter
    def groups(self, groups: dict[str, list[str]]):
        # Check groups members
        if groups is not None:
            total = [element for elements in groups.values() for element in elements]
            if self.adsorabtes.sort() != total.sort():
                raise RuntimeError("Double check group members.")

        self._groups = groups

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
        if len(energies) != len(self.data):
            raise ValueError("New adsorbate energies length doesn't match others.")

        if name in self.data.columns.values:
            raise ValueError(f"Adsorbate {name} already exists.")

        else:
            self.data[name] = energies

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

        if len(energies) != len(self.data.columns):
            raise ValueError("New sample energies length doesn't match others.")

        if name in self.data.index:
            raise ValueError(f"Sample {name} already exists.")

        else:
            self.data.loc[name] = energies

    def remove_adsorbate(self, name: str) -> None:
        """Remove an adsorbate (column).

        Args:
            name (str): The name of the adsorbate column to be removed.
        """

        self.data.drop(columns=name, inplace=True)

    def remove_sample(self, name: str) -> None:
        """Remove a sample (row).

        Args:
            name (str): The name of the sample row to be removed.
        """

        self.data.drop(index=name, inplace=True)

    def sort_data(self, targets: list[str] = ["column", "row"]) -> None:
        """Sort columns/rows of data."""

        if not set(targets) <= {"column", "row"}:
            raise ValueError(
                "Invalid target values. Should be 'column', 'row', or both."
            )

        if "column" in targets:
            self.data = self.data.sort_index(axis=1)

        if "row" in targets:
            self.data = self.data.sort_index()
