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

from typing import Optional


class Relation:
    """Describe linear scaling relations with a coefficient matrix,
    the dimensionality would be calculated on the fly. A metrics
    dict is optional but suggested.
    """

    def __init__(
        self,
        coefficients: dict[str, list[float]],
        metrics: Optional[dict[str, float]] = None,
    ) -> None:
        """Initialize Relation with coefficients.

        Args:
            coefficients (dict[str, list[float]]): Dictionary mapping
                species names to lists of coefficients.

        Raises:
            TypeError: If coefficients is not a dictionary, species name
                is not a string, or coefficients are not floats.
            ValueError: If coefficients lists have different lengths.
        """

        # Set coefficients and dim
        self._coefficients = coefficients

        # Set metrics
        self._metrics = metrics

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
        self._dim = lengths[0] - 1  # read-only

    @property
    def dim(self) -> int:
        """Dimensionality (as number of descriptors)(read-only)."""

        return self._dim

    @property
    def metrics(self) -> dict[str, float] | None:
        """Evaluation metrics (MAE/R2 or such) of this Relation."""

        return self._metrics

    @metrics.setter
    def metrics(self, metrics: dict[str, float]):
        """Set metrics, which is expect to be a 'name': 'error' dict,
        for example:
            metrics = {
                "MAE": 0.1,
                "R2": 0.2,
            }
        """
        # Check data types
        if not isinstance(metrics, dict):
            raise TypeError("metric should be a dict.")

        for value in metrics.values():
            if not isinstance(value, float):
                raise TypeError("metric values should be float.")

        self._metrics = metrics
