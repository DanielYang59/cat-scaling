# TODO: better way to initialize a Reaction,
# with energy for each species

"""Classes for representing a surface reaction.

This object is oriented towards easier recording of stoichiometric number
for surface reactions (Unlike the Reaction class from pymatgen, t), as such
species names are treated as is (for example name "*CO2" is allowed).
"""

import warnings
from typing import Any

from scaling.data.species import Species


class ReactionStep:
    """Represent a single reaction step for within a reaction."""

    def __init__(
        self,
        reactants: dict[Species, float],
        products: dict[Species, float],
    ) -> None:
        """Initialize a ReactionStep object.

        Args:
            reactants (dict[Species, float]): A dictionary representing the
                reactants of the reaction, where keys are instances of Species
                class representing the reactant species and values are floats
                representing the stoichiometric numbers.
            products (dict[Species, float]): A dictionary representing the
                products of the reaction, where keys are instances of Species
                class representing the product species and values are floats
                representing the stoichiometric numbers.

        Attributes:
            reactants (dict[Species, float]): Reactants represented as a dict
                where keys are instances of Species class representing the
                reactant species and values are floats
                representing the stoichiometric numbers.
            products (dict[Species, float]): Products represented as a dict
                where keys are instances of Species class representing the
                product species and values are floats representing the
                stoichiometric numbers.
        """

        self.reactants = reactants
        self.products = products

    def __eq__(self, other: Any) -> bool:
        """Equality comparison between two ReactionStep objects."""
        if not isinstance(other, ReactionStep):
            return False

        if (
            self.reactants == other.products
            and self.products == other.reactants
        ):
            warnings.warn("Found a reverse reaction step.")

        return (
            self.reactants == other.reactants
            and self.products == other.products
        )

    def __str__(self) -> str:
        # Add reactants
        reactants = []
        for spec, num in self.reactants.items():
            if spec.adsorbed:
                name = f"*{spec.name}"
            else:
                name = f"{spec.name}_{spec.state}"

            reactants.append(f"{num}{name}")

        # Add products
        products = []
        for spec, num in self.products.items():
            if spec.adsorbed:
                name = f"*{spec.name}"
            else:
                name = f"{spec.name}_{spec.state}"

            products.append(f"{num}{name}")

        # Assemble
        return " + ".join(reactants) + " --> " + " + ".join(products)

    def __hash__(self) -> int:
        return hash(
            (
                frozenset(self.reactants.items()),
                frozenset(self.products.items()),
            )
        )

    @property
    def reactants(self) -> dict[Species, float]:
        """Reactants represented as dict{Species: float}."""
        return self._reactants

    @reactants.setter
    def reactants(self, reactants: dict[Species, float]):
        for species, num in reactants.items():
            if not isinstance(species, Species):
                raise TypeError("Expect type Species for species.")

            if not isinstance(num, (float, int)):
                raise TypeError("Stoichiometric number should be float.")

            if num < 0:
                warnings.warn("Negative stoichiometric number found.")

        self._reactants = {k: float(v) for k, v in reactants.items()}

    @property
    def products(self) -> dict[Species, float]:
        """Products represented as dict{Species: float}."""
        return self._products

    @products.setter
    def products(self, products: dict[Species, float]):
        for species, num in products.items():
            if not isinstance(species, Species):
                raise TypeError("Expect type Species for species.")

            if not isinstance(num, (float, int)):
                raise TypeError("Stoichiometric number should be float.")

            if num < 0:
                warnings.warn("Negative stoichiometric number found.")

        self._products = {k: float(v) for k, v in products.items()}


class Reaction:
    """Represent a complete Reaction, as a collection of ReactionStep."""

    def __init__(self, reaction_steps: list[ReactionStep]) -> None:
        self.reaction_steps = reaction_steps

    def __len__(self) -> int:
        """Number of ReactionSteps."""
        return len(self.reaction_steps)

    def __getitem__(self, index):
        return self.reaction_steps[index]

    def __setitem__(self, index, value):
        self.reaction_steps[index] = value

    @property
    def reaction_steps(self) -> list[ReactionStep]:
        """Core attrib: collection of ReactionSteps."""

        return self._reaction_steps

    @reaction_steps.setter
    def reaction_steps(self, reaction_steps: list[ReactionStep]):
        if not all(isinstance(i, ReactionStep) for i in reaction_steps):
            raise TypeError("Each step should be ReactionStep.")

        if len(reaction_steps) != len(set(reaction_steps)):
            raise ValueError("Duplicate ReactionStep found.")

        self._reaction_steps = reaction_steps
