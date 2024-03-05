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

# TODO: move "ratios" from Builder to Relation

# TODO: revise positioning of properties "groups", "descriptors"

# TODO: use consistent "species" naming over "adsorbate"

import warnings
from math import isclose
from typing import Optional

import numpy as np
from sklearn.linear_model import LinearRegression

from scaling.data import Eads
from scaling.relation.relation import Relation

VALID_METHODS = {"traditional", "hybrid"}


class Builder:
    """Build scaling relation."""

    def __init__(
        self,
        data: Eads,
        descriptors: Optional[list[str]] = None,
        # ratios: Optional[dict[str, list[float]]] = None,
        groups: Optional[dict[str, list[str]]] = None,
        method: str = "traditional",
    ) -> None:
        # Check arg: data
        if not isinstance(data, Eads):
            raise TypeError("Expect data as 'Eads' type")

        self.data = data
        self.descriptors = descriptors
        # self.ratios = ratios
        self.groups = groups
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

    # @property
    # def ratios(self) -> Optional[dict[str, list[float]]]:
    #     """Mixing ratios of descriptors for each species.

    #     This property returns a dictionary where the keys are species names
    #     and the values are lists of mixing ratios corresponding to
    #     each descriptor.

    #     Returns:
    #         Optional[dict[str, list[float]]]: A dictionary mapping species
    #         names to lists of mixing ratios, None if ratios are not defined.
    #     """

    #     return self._ratios

    # @ratios.setter
    # def ratios(self, ratios: Optional[dict[str, list[float]]]):
    #     if ratios is not None:
    #         # Check if all values are lists of floats and have the same
    # length
    #         list_lengths = set()
    #         for value in ratios.values():
    #             # Check if the value is a list
    #             if not isinstance(value, list):
    #                 raise ValueError("Each value must be a list.")

    #             # Check if all elements in the list are floats
    #             if not all(isinstance(elem, float) for elem in value):
    #                 raise ValueError("Each list must contain only floats.")

    #             # Store the length of the list
    #             list_lengths.add(len(value))

    #         # Check if all ratios have the same length
    #         if len(list_lengths) > 1:
    #             raise ValueError("All lists must have the same length.")

    #     self._ratios = ratios

    @property
    def groups(self) -> Optional[dict[str, list[str]]]:
        """The groups of species, needed for traditional method.

        This property returns a dictionary where the keys are descriptor names
        and the values are lists of species belonging to each group.

        Returns:
            Optional[dict[str, list[str]]]: A dictionary where keys are
            descriptor names and values are lists of species belonging to each
            group. Returns None if groups are not defined.
        """

        return self._groups

    @groups.setter
    def groups(self, groups: Optional[dict[str, list[str]]]):
        if groups is not None:
            # Check for duplicates in each value list
            for value_list in groups.values():
                if len(set(value_list)) != len(value_list):
                    raise ValueError("Duplicate species found in group.")

        self._groups = groups

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

    def _build_composite_descriptor(
        self, spec_ratios: dict[str, float]
    ) -> np.ndarray:
        """Build a composite descriptor from child descriptors.

        This method build a composite descriptor from a list of child
        descriptors and their corresponding mixing ratios.

        Parameters:
            spec_ratios (dict[str, float]): Dict of species_name: ratio.

        Returns:
            np.ndarray: The composite descriptor as a numpy array.

        Raises:
            ValueError: If the values do not sum to 1.0 or
                if the lengths of names and ratios do not match.
        """

        # Check arg: spec_ratios
        if not isclose(sum(spec_ratios.values()), 1.0, abs_tol=1e-04):
            raise ValueError("Ratios should sum to 1.0.")

        # Fetch child descriptors
        child_descriptors = np.array(
            [
                self.data.get_adsorbate(species)
                for species in spec_ratios.keys()
            ]
        )

        # Construct composite descriptor (from child descriptors)
        return np.sum(
            child_descriptors
            * np.array(list(spec_ratios.values()))[:, np.newaxis],
            axis=0,
        )

    def _builder(
        self, spec_name: str, ratios: dict[str, float]
    ) -> tuple[list[float], float, float]:
        """Core worker for building scaling relations.

        Parameters:
        spec_name (str): name of the target species to build.
        ratios (dict[str, float]): Dict of species_name: ratio.

        Returns:
            coefs (list[float]): coefficients correspond to each descriptor
            intercept (float): intercept
            metrics (float): metrics

        Raises:
            ValueError: If the ratios do not sum to 1.0 or if the lengths of
                descriptors and ratios do not match.

        How this works:
            1. Build composite descriptor:
            First a composite descriptor is built for actual linear
            regression process. For example there may be two nominated
            descriptors (each as a np.ndarray) and their mixing ratios
            (for example [0.2, 0.8]). Then the composite descriptor is
            built as:
                comp_des = 0.2 * descriptor_A + 0.8 * descriptor_B

            2. Perform linear regression:
            The composite descriptor would be used to perform linear
            regressions for each adsorbate (target), where there would
            be a coefficient, an intercept and a metrics score:
                coef, intercept, score = LinearRegression(*)

            3. Map scaling coefficients to original descriptors:
            As the linear regression is construction upon the composite
            descriptor, we need to map the scaling parameters to
            the origin descriptors, which is straightforward:
                Multiple the coefficients with corresponding ratios,
                and leave the intercept unchanged.

        In a more "mathematical" manner:
            (where comp_des, A, N and target are arrays)
            TODO: format this formula block (latex?)
            1. The composite descriptor:
                comp_des = cof_A * A + ... + cof_N * N

            2. After a linear regression:
                target = slope * comp_des + intercept
        """

        # Build composite descriptor
        composite_descriptor = self._build_composite_descriptor(ratios)

        # Perform linear regression
        _comp_des = composite_descriptor.reshape(-1, 1)
        _target = self.data.get_adsorbate(spec_name).reshape(-1, 1)

        reg = LinearRegression().fit(_comp_des, _target)

        # Collect results
        # Map scaling coefficients to original descriptors
        # As there is only the composite descriptor,
        # there should be only one coefficient each species
        coefs = [float(reg.coef_[0] * ratio) for ratio in ratios.values()]
        intercept = float(reg.intercept_[0])

        metrics = reg.score(_comp_des, _target)

        assert len(coefs) == len(ratios)
        return coefs, intercept, metrics

    def build_traditional(
        self,
        groups: dict[str, list[str]],
    ) -> Relation:
        """Build scaling relations the traditional way, where each species is
        approximated by a single descriptor within each group.

        Parameters:
            groups (dict[str, list[str]]): A dictionary where keys represent
                descriptor names and values represent lists of species
                associated with that descriptor.

        Returns:
            Relation: A Relation object containing coefficients and metrics
                of the scaling relations.
        """

        # Build for each species
        coefficients_dict = {}
        metrics_dict = {}

        for descriptor, species in groups.items():
            for spec_name in species:
                coefs, intercept, metrics = self._builder(
                    spec_name=spec_name,
                    ratios={descriptor: 1.0},
                )

                # Collect intercept into coefficients
                # TODO: separate intercept from coefficients in Relation
                coefs.append(intercept)

                # Collect results
                coefficients_dict[spec_name] = coefs
                metrics_dict[spec_name] = metrics

        return Relation(coefficients_dict, metrics_dict)

    # def build_hybrid(self) -> Relation:
    #     pass
