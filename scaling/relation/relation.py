# TODO: unit test needs update to test properties
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


from math import isclose
from typing import Optional


class Relation:
    """Describe linear scaling relations with a coefficient matrix,
    the dimensionality would be calculated on the fly. A metrics
    dict is optional but suggested.
    """

    def __init__(
        self,
        coefficients: dict[str, list[float]],
        intercepts: dict[str, float],
        metrics: Optional[dict[str, float]] = None,
        ratios: Optional[dict[str, dict[str, float]]] = None,
    ) -> None:
        """Initialize Relation with coefficients.

        Args:
            coefficients (dict[str, list[float]]): Dictionary mapping
                species names to lists of coefficients.
            intercepts (dict[str, float]): Dictionary mapping
                species names to intercept.
            metrics (Optional[dict[str, float]]): Evaluation metrics
                (MAE/R2 or such) of this Relation. Defaults to None.
            ratios (Optional[dict[str, dict[str, float]]]): The keys
                are species names and the values are lists of
                mixing ratios corresponding to each descriptor.
        """

        # Set properties
        self.coefficients = coefficients
        self.intercepts = intercepts
        self.metrics = metrics
        self.ratios = ratios

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
        """Dimensionality (as number of descriptors)(read-only)."""

        return len(next(iter(self.coefficients.values())))

    @property
    def metrics(self) -> Optional[dict[str, float]]:
        """Evaluation metrics (MAE/R2 or such) of this Relation."""

        return self._metrics

    @metrics.setter
    def metrics(self, metrics: Optional[dict[str, float]]):
        """Set metrics, which is expect to be a "species: error" dict,
        for example:
            metrics = {
                "*CO": 0.8,
                "*OH": 0.9,
            }
        """

        # Check data types
        if metrics is not None:
            if not isinstance(metrics, dict):
                raise TypeError("metric should be a dict.")

            for value in metrics.values():
                if not isinstance(value, float):
                    raise TypeError("metric values should be float.")

            self._metrics = metrics

    @property
    def ratios(self) -> Optional[dict[str, dict[str, float]]]:
        """Mixing ratios of descriptors for each species.

        Returns:
            Optional[dict[str, dict[float]]]: a dictionary where the keys
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
    def ratios(self, ratios: Optional[dict[str, dict[str, float]]]):
        if ratios is not None:
            # Check if all ratio are dict and have the same length
            dict_lengths = set()
            for ratio_dict in ratios.values():
                # Check if the ratio_dict is a dict
                if not isinstance(ratio_dict, dict):
                    raise ValueError("Each ratio_dict must be a dict.")

                # Check if ratios sum to one
                if not isclose(sum(ratio_dict.values()), 1.0, abs_tol=0.01):
                    raise ValueError(
                        "Ratios for each species should sum to one."
                    )

                # Store the length of the dict
                dict_lengths.add(len(ratio_dict))

            # Check if ratios dict have the same length
            if len(dict_lengths) > 1:
                raise ValueError("Ratio dict must have the same length.")

        self._ratios = ratios
