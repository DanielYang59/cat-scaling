"""Build scaling relations (Relation) from adsorption energy (Eads).

The following would take CO2 reduction to CH4 reaction (CO2RR) as an example.

The traditional method:
    As proposed in Peterson and Nørskov's work titled *Activity Descriptors for
    CO2 Electroreduction to Methane on Transition-Metal Catalysts*, involves
    grouping species involved in the CO2RR process into two categories:
    C-centered (*COOH, *CO, *CHO, *CH2O) and O-centered (*OCH3, *O, *OH).

    Within each group, a descriptor is nominated (in the original work, it was
    *CO and *OH respectively). The adsorption energies of other species can
    then be approximated by their respective descriptor, effectively reducing
    the dimensionality of the problem from 7D to 2D.
    This reduction enables visualization and simplifies the analysis process.

The hybrid method:
    As discussed in my MPhil work (at https://arxiv.org/abs/2402.03876),
    recognizes that many species involved in the process include both oxygen
    and carbon. Therefore, it makes sense to include descriptors for both
    elements in the relation. In that work, I discovered that this simple
    approach can improve the coefficient of determination (R2) by
    approximately 0.06-0.13, thus confirming its validity.

    Moreover, this method doesn't incur significant costs:
        - Confirming the descriptor ratio requires only a few linear fittings.
        - Users are not required to specify the descriptor,
            though they have the option to do so.
        - The resulting Relation is compatible with the traditional method.
"""

from scaling.data import Eads
from scaling.relation import Relation

VALID_METHODS = {"traditional", "hybrid"}


class Builder:
    def __init__(
        self,
        data: Eads,
        method: str = "traditional",
    ) -> None:
        # Check arg: data
        if not isinstance(data, Eads):
            raise TypeError("Expect data as 'Eads' type")

        self.data = data
        self.method = method

    @property
    def method(self) -> str:
        """Method used to build the Relation.
        Current supported methods:
            traditional: Use a fix descriptor ratio
            hybrid: Dynamically determine optimal mixing ratio

        TODO: this docstring could be made more detailed
        """

        return self._method

    @method.setter
    def method(self, method: str):
        # Check method validity
        if method.lower() not in VALID_METHODS:
            raise ValueError(
                f"Unsupported build method, support: {VALID_METHODS}"
            )

        self._method = method.lower()

    def build_traditional(self) -> Relation:
        pass

    def build_hybrid(self) -> Relation:
        # Also need to return mixing ratio
        pass
