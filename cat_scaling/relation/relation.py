# NOTE (need consider): EadsRelation does NOT carry complete information:
# descriptors are missing, should info be duplicate?

# TODO: unit test needs update to test properties for EadsRelation

# TODO: unit test completely missing for DeltaERelation

# TODO: module docstring needs update (or move to separate docs)

"""Describe linear scaling relations with a coefficient matrix.

Linear scaling relations describe the adsorption energy of a species by a
linear combination of several other species (chosen as descriptors),
akin to dimension reduction methods in machine learning.

For any species Z, its adsorption energy EadsZ could be expressed as:
    EadsZ = aZ * EadsX + bZ * EadsY + cZ

where X and Y are two nominated descriptors, and aZ/bZ/cZ are coefficients
specific to Z. While you can select an arbitrary number of descriptors,
for 3D visualization, it's advisable to limit the number below three.

Thus, it's convenient to represent a linear scaling relation with a
coefficient matrix, similar to solving systems of equations in linear algebra.

Coefficient matrix:
               Descriptor_X       Descriptor_Y   .......    Constant
    EadsI   =  aI                 bI             .......    cI
    EadsII  =  aII                bII            .......    cII
    ...
    Eads X  =  aX                 bX             .......    cX
"""

from __future__ import annotations

import warnings
from math import isclose

import numpy as np


class EadsRelation:
    """Describe linear scaling relations with a coefficient matrix.

    Attributes:
        coefficients (dict[str, list[float]]): Dict mapping
            species names to coefficients.
        intercepts (dict[str, float]): Dict mapping
            species names to intercept.
        metrics (dict[str, float]): Evaluation metrics
            (MAE/R2 or such) of this Relation, for each species.
        ratios (dict[str, dict[str, float]]): The keys
            are species names and the values are lists of
            mixing ratios corresponding to each descriptor.
        dim (int): Dimensionality as the number of descriptors.
    """

    def __init__(
        self,
        coefficients: dict[str, list[float]],
        intercepts: dict[str, float],
        metrics: dict[str, float],
        ratios: dict[str, dict[str, float]],
    ) -> None:
        """Initialize with coefficients and intercepts."""

        # Set properties
        self.coefficients = coefficients
        self.intercepts = intercepts
        self.metrics = metrics
        self.ratios = ratios

    def __str__(self) -> str:
        """String representation of the Relation."""

        # Add descriptors
        string = f"Descriptors: {', '.join(self.coefficients.keys())}\n\n"

        # Add coefficient matrix header
        string += (
            f"{'Adsorbate':<10}"
            + "".join([f"coef_{n:<6}" for n in range(self.dim)])
            + "intercept\n"
        )

        # Add coefficient matrix and intercept
        for ads in self.coefficients.keys():
            string += f"{ads:<10}"
            string += " ".join([f"{i:<10.2f}" for i in self.coefficients[ads]])
            string += f"{self.intercepts[ads]:<10.2f}\n"

        # Add metrics
        string += "\nAdsorbate Metrics\n"
        for name, metric in self.metrics.items():
            string += f"{name:<10}{metric:<10.2f}\n"

        # Add ratios
        string += "\nAdsorbate Ratios\n"
        for name, ratios in self.ratios.items():
            string += f"{name:<10}{ratios}\n"

        return string

    @property
    def coefficients(self) -> dict[str, list[float]]:
        """Coefficient matrix."""

        return self._coefficients

    @coefficients.setter
    def coefficients(self, coefficients: dict[str, list[float]]):
        """Set coefficients and calculate dim."""

        # Check data types
        if not isinstance(coefficients, dict):
            raise TypeError("Coefficients must be a dictionary")

        for key, value in coefficients.items():
            if not isinstance(key, str):
                raise TypeError("Species name must be strings")
            if not isinstance(value, list):
                raise TypeError("Input coefficients must be lists")
            for item in value:
                if not isinstance(item, float):
                    raise TypeError("Coefficients must be floats")

        # Check coefficient length consistency
        lengths = [len(value) for value in coefficients.values()]
        if len(set(lengths)) > 1:
            raise ValueError("All coefficients must have the same length")

        self._coefficients = coefficients

    @property
    def intercepts(self) -> dict[str, float]:
        """Intercept of scaling relation."""
        return self._intercepts

    @intercepts.setter
    def intercepts(self, intercepts: dict[str, float]):
        if not isinstance(intercepts, dict):
            raise TypeError("intercepts should be a dict.")

        if not all(
            isinstance(value, (int, float)) for value in intercepts.values()
        ):
            raise TypeError("intercept value should be float.")

        self._intercepts = {
            key: float(value) for key, value in intercepts.items()
        }

    @property
    def dim(self) -> int:
        """Dimensionality (as number of descriptors)."""

        return len(next(iter(self.coefficients.values())))

    @property
    def metrics(self) -> dict[str, float]:
        """Evaluation metrics (MAE/R2 or such) of this Relation."""

        return self._metrics

    @metrics.setter
    def metrics(self, metrics: dict[str, float], warn_threshold: float = 0.5):
        """Set metrics.

        Parameters:
            metrics (dict[str, float]): A dictionary representing metrics with
                keys as species and values as error values.
                For example:
                    metrics = {
                        "*CO": 0.8,
                        "*OH": 0.9,
                    }
            warn_threshold (float, optional): The threshold below which a
                warning will be issued for low metrics. Defaults to 0.5.

        Raises:
            TypeError: If metrics is not a dictionary or
                metric values are not floats.

        Warns:
            UserWarning: If any metric value is below the warn_threshold.
        """

        # Check data types
        if not isinstance(metrics, dict):
            raise TypeError("metric should be a dict.")

        for name, value in metrics.items():
            if not isinstance(value, float):
                raise TypeError("metric values should be float.")
            if value < warn_threshold:
                warnings.warn(f"Low metrics for {name} at {value}.")

        self._metrics = metrics

    @property
    def ratios(self) -> dict[str, dict[str, float]]:
        """Mixing ratios of descriptors for each species.

        Returns:
            dict[str, dict[float]]: a dictionary where the keys
                are species names and the values are dict of
                mixing ratios.

        Example:
            ratios = {
                "*COOH": {"*CO": 0.25, "*OH": 0.75},
                }
            means for species *COOH, two descriptors *CO and *OH are used,
            and their ratios are 0.25 and 0.75 respectively.
        """

        return self._ratios

    @ratios.setter
    def ratios(self, ratios: dict[str, dict[str, float]]):
        # Check if all ratio are dict and have the same length
        dict_lengths = set()
        for ratio_dict in ratios.values():
            # Check if the ratio_dict is a dict
            if not isinstance(ratio_dict, dict):
                raise ValueError("Each ratio_dict must be a dict.")

            # Check if ratios sum to one
            if not isclose(sum(ratio_dict.values()), 1.0, abs_tol=0.01):
                raise ValueError("Ratios for each species should sum to one.")

            # Store the length of the dict
            dict_lengths.add(len(ratio_dict))

        # Check if ratios dict have the same length
        if len(dict_lengths) > 1:
            raise ValueError("Ratio dict must have the same length.")

        self._ratios = ratios


class DeltaERelation:
    """Class for describing reaction (free) energy Relation,
    by a list of coefficient arrays, each correspond to a reaction step.

    Args:
        coefficients (list[np.ndarray]): List of coefficient arrays
            representing the energy change (DeltaE) scaling relation.
            Each array corresponds to a single reaction step.

    Attributes:
        coefficients (list[np.ndarray]): List of coefficient arrays
            representing the energy change (DeltaE) scaling relation.
            Each array corresponds to a reaction step.
    """

    def __init__(
        self,
        coefficients: list[np.ndarray],
    ) -> None:
        self.coefficients = coefficients

    @property
    def coefficients(self) -> list[np.ndarray]:
        """Energy change (DeltaE) scaling relation represented by a
        list of coefficient arrays, each correspond to a reaction step.
        """

        return self._coefficients

    @coefficients.setter
    def coefficients(self, coefficients: list[np.ndarray]):
        # Check if coefficients is a list
        if not isinstance(coefficients, list):
            raise TypeError("Coefficients should be a list.")

        # Check if all elements in the list are numpy arrays
        if not all(isinstance(arr, np.ndarray) for arr in coefficients):
            raise TypeError("All coefficients should be numpy arrays.")

        # Check if all arrays have the same length
        if len(set(arr.shape[0] for arr in coefficients)) > 1:
            raise ValueError(
                "All coefficient arrays should have the same length."
            )

        self._coefficients = coefficients

    @property
    def dim(self) -> int:
        """Dimensionality (as number of descriptors)."""

        return len(self.coefficients[0]) - 1  # 1 for intercept

    def eval_limit_potential_2D(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate limiting potential on a 2D grid.

        Args:
            x (np.ndarray): 1D array representing the x-coordinates.
            y (np.ndarray): 1D array representing the y-coordinates.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - limit_potential (np.ndarray): 2D array containing the
                    negative of the maximum delta E (limiting potential).
                - step_indexes (np.ndarray): 2D array containing the indexes
                    of the reaction step corresponding to
                    the maximum delta E values (rate-determining step).
        """

        def _eval_deltaE_2D(
            x: np.ndarray, y: np.ndarray, coef: np.ndarray
        ) -> np.ndarray:
            """Evaluate delta E for a single reaction step, on a 2D grid.

            Args:
                x (np.ndarray): 1D array representing the x-coordinates.
                y (np.ndarray): 1D array representing the y-coordinates.
                coef (np.ndarray): Array of coefficients
                    [x_coef, y_coef, intercept].

            Returns:
                np.ndarray: 2D array containing delta E values.
            """

            # Generate 2D mesh
            xx, yy = np.meshgrid(x, y, indexing="xy")

            # Extract coefficients
            coef_x, coef_y, intercept = coef

            # TODO: more efficient method/algorithm?
            return xx * coef_x + yy * coef_y + intercept

        def _eval_limit_potential_2D(
            deltaEs: list[np.ndarray],
        ) -> tuple[np.ndarray, np.ndarray]:
            """Evaluate limiting potential on a 2D grid.

            Args:
                deltaEs (List[np.ndarray]): List of 2D arrays containing
                    delta E values for each reaction step.

            Returns:
                Tuple[np.ndarray, np.ndarray]: A tuple containing:
                    - limit_potential (np.ndarray): 2D array containing
                        the negative of the maximum delta E
                        (limiting potential).
                    - step_indexes (np.ndarray): 2D array containing the
                        indexes of the reaction step corresponding to
                        the maximum delta E values.
            """

            # Stack 2D deltaE meshes
            deltaE_stack = np.stack(deltaEs, axis=2)

            # Find maximum values and corresponding indexes
            # NOTE: a negative sign is added by definition
            limit_potential = -np.amax(deltaE_stack, axis=2)
            step_indexes = np.argmax(deltaE_stack, axis=2)

            # Internal sanity check of limiting potential
            assert np.all(limit_potential <= 0), "Internal error: U_L"

            return limit_potential, step_indexes

        # Check coefs
        if self.dim != 2:
            raise ValueError("Expect a Relation with two descriptors.")

        # Check base arrays
        if not (isinstance(x, np.ndarray) and x.ndim == 1):
            raise TypeError("Expect 1D x array.")
        if not (isinstance(y, np.ndarray) and y.ndim == 1):
            raise TypeError("Expect 1D y array.")

        # Calculate delta E for each reaction step
        deltaEs = []
        for coef in self.coefficients:
            deltaEs.append(_eval_deltaE_2D(x, y, coef))

        # Calculate limiting potential and corresponding step index
        return _eval_limit_potential_2D(deltaEs)
