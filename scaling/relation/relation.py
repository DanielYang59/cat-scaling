"""Module for describing linear scaling relations with a coefficient matrix.

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


class Relation:
    """Describe linear scaling relations with a coefficient matrix."""
    def __init__(
        self,
        coefficients: dict[str, list[float]],
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
        # Check and take coefficients
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

        lengths = [len(value) for value in coefficients.values()]
        if len(set(lengths)) > 1:
            raise ValueError("All coefficients must have the same length")

        self.coefficients = coefficients

        # Calculate dimensionality (as number of descriptors)
        self.dim = lengths[0] - 1
