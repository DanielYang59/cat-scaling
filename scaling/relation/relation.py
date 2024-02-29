# TODO: unit test needs update to test "metrics" and "descriptors"
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


import warnings
from typing import Optional


class Relation:
    """Describe linear scaling relations with a coefficient matrix,
    the dimensionality would be calculated on the fly. A metrics
    dict is optional but suggested.
    """

    def __init__(
        self,
        coefficients: dict[str, list[float]],
        descriptors: Optional[list[str]] = None,
        metrics: Optional[dict[str, float]] = None,
    ) -> None:
        """Initialize Relation with coefficients.

        Args:
            coefficients (dict[str, list[float]]): Dictionary mapping
                species names to lists of coefficients.
            descriptors (Optional[list[str]]): List of descriptors used
                for scaling relations. Defaults to None.
            metrics (Optional[dict[str, float]]): Evaluation metrics
                (MAE/R2 or such) of this Relation. Defaults to None.
        """

        # Set properties
        self.coefficients = coefficients
        self.descriptors = descriptors
        self.metrics = metrics

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
    def descriptors(self) -> Optional[list[str]]:
        """Descriptors used for scaling relations."""

        return self._descriptors

    @descriptors.setter
    def descriptors(self, descriptors: Optional[list[str]]):
        if descriptors is not None:
            if not isinstance(descriptors, list):
                raise TypeError("Descriptors must be a list of strings.")

            if not all(isinstance(desc, str) for desc in descriptors):
                raise TypeError("Each descriptor must be a string.")

            if len(descriptors) != len(set(descriptors)):
                raise ValueError("Duplicate descriptors are not allowed.")

            if len(descriptors) > 2:
                warnings.warn(f"Got {len(descriptors)} descriptors.")

        self._descriptors = descriptors

    @property
    def dim(self) -> int:
        """Dimensionality (as number of descriptors)(read-only)."""

        return len(next(iter(self.coefficients.values()))) - 1

    @property
    def metrics(self) -> Optional[dict[str, float]]:
        """Evaluation metrics (MAE/R2 or such) of this Relation."""

        return self._metrics

    @metrics.setter
    def metrics(self, metrics: Optional[dict[str, float]]):
        """Set metrics, which is expect to be a "name: error" dict,
        for example:
            metrics = {
                "MAE": 0.1,
                "R2": 0.2,
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
