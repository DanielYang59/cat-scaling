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
        - Confirming the descriptor ratio requires only few linear regressions.
        - Users are not required to specify the descriptor,
            though they have the option to do so.
        - The resulting Relation is compatible with the traditional method.
"""

# TODO: set "ratios" as property
# TODO: use consistent "adsorbate" and "species"

import warnings
from math import isclose
from typing import Optional

import numpy as np
from sklearn.linear_model import LinearRegression

from scaling.data import Eads
from scaling.relation import Relation

VALID_METHODS = {"traditional", "hybrid"}


class Builder:
    """Build scaling relation."""

    def __init__(
        self,
        data: Eads,
        descriptors: Optional[list[str]] = None,
        method: str = "traditional",
    ) -> None:
        # Check arg: data
        if not isinstance(data, Eads):
            raise TypeError("Expect data as 'Eads' type")

        self.data = data
        self.descriptors = descriptors
        self.method = method

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
    def method(self) -> str:
        """Method used to build the Relation.
        Current supported methods:
            traditional: Use a fix descriptor ratio
            hybrid: Dynamically determine optimal mixing ratio
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

    def _builder(
        self,
        descriptors: list[str],
        ratios: list[float],
    ) -> Relation:
        """Core worker for building scaling relations.

        Parameters:
        descriptors (list[str]): Descriptor names.
        ratios (list[float]): Mixing ratios corresponding
            to descriptors.

        Returns:
            Relation: An instance of Relation containing scaling coefficients
                and metrics.

        Raises:
            ValueError: If the ratios do not sum to 1.0 or if the lengths of
                descriptors and ratios do not match.

        How this builder works:
            1. Construct composite descriptor:
            First a composite descriptor is constructed for actual linear
            regression process. For example there may be two nominated
            descriptors (each as a np.ndarray) and their mixing ratios
            (for example [0.2, 0.8]). Then the composite descriptor is
            constructed as:
                comp_descriptor = 0.2 * descriptor_A + 0.8 * descriptor_B

            2. Perform linear regression:
            The composite descriptor would be used to perform linear
            regressions with each adsorbate. For each adsorbate, there
            would be a coefficient, an intercept and a metrics score:
                coef, intercept, score = LinearRegression(*)

            3. Map scaling coefficients to original descriptors:
            As the linear regression is construction upon the composite
            descriptor, we need to map the scaling parameters to
            the origin descriptors, which is straightforward:
                Multiple the coefficients with corresponding ratios,
                and leave the intercept unchanged.
        """

        # Check arg: ratios
        if not isclose(sum(ratios), 1.0, abs_tol=1e-04):
            raise ValueError("Ratios should sum to 1.0.")

        # Check descriptors and ratios length
        if len(descriptors) != len(ratios):
            raise ValueError("Descriptors and ratios length mismatch.")

        # Fetch child descriptors
        child_descriptors = np.array(
            [self.data.get_adsorbate(species) for species in descriptors]
        )

        # Construct composite descriptor (from child descriptors)
        composite_descriptor = np.sum(
            child_descriptors * np.array(ratios)[:, np.newaxis], axis=0
        )

        # Perform linear regressions for each species
        coefficients = {}
        metrics = {}

        for species in self.data.adsorbates:
            # Perform linear regression
            _comp_des = composite_descriptor
            _target = self.data.get_adsorbate(species)
            reg = LinearRegression().fit(_comp_des, _target)

            # Map scaling coefficients to original descriptors
            _coefs = [reg.coef_ * ratio for ratio in ratios]
            _coefs.append(reg.intercept_)  # append intercept

            # Collect final results
            coefficients[species] = _coefs
            metrics[species] = reg.score(_comp_des, _target)

        # Build Relation
        return Relation(coefficients, metrics)

    # def build_traditional(self, groups: []) -> Relation:

    #     pass

    # def build_hybrid(self) -> Relation:
    #     # NOTE: Also need to return (set) mixing ratio
    #     pass
