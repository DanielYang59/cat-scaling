# TODO: add add_correction method
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

import copy

import numpy as np

from scaling.data.reaction import Reaction
from scaling.relation.relation import EadsRelation


class AdsEToDeltaE:
    """Convert adsorption energy scaling Relation to
    reaction energy Relation, based on a Reaction. Would also need
    free species energies. Could optionally add corrections terms
    (zero-point energies, solvation energies or such).

    The resulted coefficient matrix would have one set of
    coefficient for each reaction step. # TODO
    """

    def __init__(
        self,
        relation: EadsRelation,
        reaction: Reaction,
        species_energies: dict[str, float],
    ) -> None:
        self.relation = relation
        self.reaction = reaction
        self.species_energies = species_energies

    def _convert(self) -> list[float]:
        """Convert adsorption energy Relation to reaction energy change
        Relation for a single reaction step.
        """

        # Initialize an empty array to host coefficients and intercept
        coef_array = np.zeros(self.relation.dim + 1)

        # Convert the reactants side
        for spec, num in self.reaction.reactants.items():
            # Compile coefficients and intercept as a single array in
            # form: [coef_0, coef_1, ..., coef_n, intercept]
            temp_arr = copy.deepcopy(self.relation.coefficients[spec.name])
            temp_arr.append(self.relation.intercepts[spec.name])

            # Add free species energy for adsorbed species to constant
            # (intercept) term. For example, for species "*CO2", the energy
            # for free "CO2" should be added
            if spec.adsorbed:
                if spec.energy is None:
                    raise ValueError("Missing energy for adsorbed species.")
                temp_arr[-1] += spec.energy

            # CAUTION: a minus sign is needed for reactants
            coef_array -= num * np.array(temp_arr)

        # Convert the products side
        for spec, num in self.reaction.products.items():
            temp_arr = copy.deepcopy(self.relation.coefficients[spec.name])
            temp_arr.append(self.relation.intercepts[spec.name])

            # Add free species energy for adsorbed species
            if spec.adsorbed:
                if spec.energy is None:
                    raise ValueError("Missing energy for adsorbed species.")
                temp_arr[-1] += spec.energy

            coef_array += num * np.array(temp_arr)

        return coef_array
