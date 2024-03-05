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

The adaptive method:
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

# TODO: use consistent naming for "species" and "adsorbate" , where "adsorbate"
# is more specific and "species" is more general (which might be confusing
# together with "sample")

# TODO: builder should skip descriptors themselves

import warnings
from math import isclose

import numpy as np
from sklearn.linear_model import LinearRegression

from scaling.data import Eads
from scaling.relation.relation import Relation


class Builder:
    """Build scaling relation."""

    def __init__(
        self,
        data: Eads,
    ) -> None:
        # Check arg: data
        if not isinstance(data, Eads):
            raise TypeError("Expect data as 'Eads' type")

        self.data = data

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
        coefs = [float(reg.coef_[0][0] * ratio) for ratio in ratios.values()]
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

        coefficients_dict = {}
        intercepts_dict = {}
        metrics_dict = {}
        ratios_dict = {}

        for descriptor, species in groups.items():
            for spec_name in species:
                # Build for each species
                ratios = {descriptor: 1.0}  # single descriptor

                coefs, intercept, metrics = self._builder(
                    spec_name=spec_name,
                    ratios=ratios,
                )

                # Collect results
                coefficients_dict[spec_name] = coefs
                intercepts_dict[spec_name] = intercept
                metrics_dict[spec_name] = metrics
                ratios_dict[spec_name] = ratios

        return Relation(
            coefficients_dict, intercepts_dict, metrics_dict, ratios_dict
        )

    def build_adaptive(
        self, descriptors: list[str], step_length: float = 1.0
    ) -> Relation:
        """Build scaling relations with descriptors ratios determined on
        the fly.

        Parameters:
            descriptors (list[str, str]): A list of the two descriptor names
                for the scaling relations.
            step_length (float, optional): A percentage value indicating the
                step size for searching the optimal ratio. Defaults to 1.0.

        Returns:
            Relation: A Relation object containing coefficients, intercepts,
                metrics, and the optimal ratios for the descriptors.

        Raises:
            ValueError: If the step_length is not a float or int, or if it's
                not within the range (0, 100). If the number of descriptors
                provided is not 2, or if there are duplicate descriptors.

        Warnings:
            If the step_length is greater than 5, a warning is issued
            indicating that a large step length may harm accuracy.

            If the step_length is less than 0.1, a warning is issued
            indicating that a small step length may slow down searching.
        """

        # Check arg: step_length
        if not isinstance(step_length, (float, int)) and 0 < step_length < 100:
            raise ValueError(f"Illegal step length {step_length}.")

        if step_length > 5:
            warnings.warn("Large step length may harm accuracy.")

        elif step_length < 0.1:
            warnings.warn("Small step length may slow down searching.")

        # Convert step_length to percentage
        step_length = step_length / 100

        # Check arg: descriptors
        if len(descriptors) != 2:
            raise ValueError("Expect two descriptors for adaptive method.")
        if len(descriptors) != len(set(descriptors)):
            raise ValueError("Duplicate descriptors not allowed.")

        coefficients_dict = {}
        intercepts_dict = {}
        metrics_dict = {}
        ratios_dict = {}

        # Iterate over each adsorbate
        for ads in self.data.adsorbates:
            # Determine an optimal descriptor mixing ratio
            scores = {}

            for ratio in np.arange(0, 1 + step_length, step_length):
                ratios = {
                    descriptors[0]: ratio,
                    descriptors[1]: 1 - ratio,
                }

                _coefs, _intercept, metrics = self._builder(
                    spec_name=ads,
                    ratios=ratios,
                )

                scores[ratio] = metrics

            # Rerun regression with the optimal ratio
            opt_ratio = max(scores, key=lambda k: scores[k])

            opt_ratios = {
                descriptors[0]: opt_ratio,
                descriptors[1]: 1 - opt_ratio,
            }

            coefs, intercept, metrics = self._builder(
                spec_name=ads,
                ratios=opt_ratios,
            )

            # Collect results
            coefficients_dict[ads] = coefs
            intercepts_dict[ads] = intercept
            metrics_dict[ads] = metrics
            ratios_dict[ads] = opt_ratios

        return Relation(
            coefficients_dict, intercepts_dict, metrics_dict, ratios_dict
        )
