"""Work on top of pre-build Relation to perform chemically
meaningful analysis: limiting potential, selectivity, and
prepare for plotters.

How this works:
TODO: format the following formula
Consider an example surface reaction:
    *A + B = *C + D
where '*' denotes an active site, and B/D represent free molecules.

The energy change (ΔE) for this reaction can be calculated as:
    ΔE = E_products - E_reactants

In this specific example:
    E_reactants = E_*A + E_B = E_A + E_B + E_ads_A + E*
    E_products = E_*C + E_D = E_C + E_D + E_ads_C + E*

Thus, the reaction energies are converted to adsorption energies,
with the energies of the free species being constant.

Therefore, to convert an EadsRelation to a DeltaERelation,
we simply need to add additional terms for the free species.

Also more correction terms (ZPE, solvation or such) could also be included.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from scaling.data.reaction import Reaction, ReactionStep
from scaling.relation.relation import DeltaERelation

if TYPE_CHECKING:
    from scaling.relation.relation import EadsRelation


class AdsorbToDeltaE:
    """Convert adsorption energy scaling Relation to
    reaction energy change Relation, based on a given Reaction.

    Need free species energies, and could optionally add
    corrections terms (zero-point energies, solvation energies or such).

    The resulted DeltaERelation would have one set of
    coefficient for each reaction step.
    """

    def __init__(
        self,
        relation: EadsRelation,
        reaction: Reaction,
    ) -> None:
        # Set attribs
        self.relation = relation
        self.reaction = reaction

    def _convert_step(self, step: ReactionStep) -> np.ndarray:
        """Convert adsorption energy Relation to reaction energy change
        Relation for a single reaction step.
        """

        # Initialize an empty array to store coefficients and intercept
        # Need number of descriptors + 1 (1 for intercept)
        coef_array = np.zeros(self.relation.dim + 1)

        # Convert the reactants side
        for spec, num in step.reactants.items():
            # Pack coefficients and intercept as a single array:
            # [coef_0, coef_1, ..., coef_n, intercept]
            if spec.adsorbed:
                complete_name = f"*{spec.name}"  # "CO2" -> "*CO2"

                spec_arr = copy.copy(self.relation.coefficients[complete_name])
                spec_arr.append(self.relation.intercepts[complete_name])

            else:
                # For free species/molecules, coefs are zero
                spec_arr = np.zeros(self.relation.dim + 1)

            # Add free species energy for adsorbed species to constant
            # (intercept) term.
            # NOTE: For species "*CO2", the energy for free "CO2"
            # should be added (provided)
            spec_arr[-1] += spec.energy

            # Add correction term
            spec_arr[-1] += spec.correction

            # NOTE: a minus sign is needed for reactants
            coef_array -= num * np.array(spec_arr)

        # Convert the products side
        for spec, num in step.products.items():
            if spec.adsorbed:
                complete_name = f"*{spec.name}"
                spec_arr = copy.copy(self.relation.coefficients[complete_name])
                spec_arr.append(self.relation.intercepts[complete_name])

                spec_arr = copy.copy(self.relation.coefficients[complete_name])
                spec_arr.append(self.relation.intercepts[complete_name])
            else:
                spec_arr = np.zeros(self.relation.dim + 1)

            # Add energy
            spec_arr[-1] += spec.energy

            # Add correction
            spec_arr[-1] += spec.correction

            coef_array += num * np.array(spec_arr)

        return coef_array

    def convert(self) -> DeltaERelation:
        """Convert a list of ReactionStep to energy change Relation."""

        # Work on each step
        coefs = [self._convert_step(step) for step in self.reaction.steps]

        # Build energy change relation (DeltaERelation)
        return DeltaERelation(coefficients=coefs)
