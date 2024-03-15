"""Class for representing a surface Reaction.

This object is oriented towards easier recording of stoichiometric number
for surface reactions (Unlike the Reaction class from pymatgen), as such
species names are treated as is (for example name like "*CO2" is allowed).
"""

from __future__ import annotations

import re
import warnings
from typing import Any

from scaling.data.species import Species


class ReactionStep:
    """Represent a single reaction step within a Reaction."""

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
                name = spec.name

            reactants.append(f"{num}{name}")

        # Add products
        products = []
        for spec, num in self.products.items():
            if spec.adsorbed:
                name = f"*{spec.name}"
            else:
                name = spec.name

            products.append(f"{num}{name}")

        # Assemble
        return " + ".join(reactants) + " -> " + " + ".join(products)

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

    @staticmethod
    def _sepa_stoi_number(name: str) -> tuple[float, str]:
        """Separate species name to (stoichiometric_number, name).

        Species without stoichiometric numbers would be treated
        as stoichiometric numbers being 1.0.

        Examples:
            "*CO2(-1, 0)" -> (1.0, "*CO2(-1, 0)")
            "2H2O_g(-2, -3)" -> (2.0, "H2O_g(-2, -3)")
        """
        name = name.strip()

        # Use re to separate leading digits and name
        match = re.match(r"^(\d+(\.\d+)?)(.*)$", name)
        if match:
            stoi_number_str = match.group(1)
            species_name = match.group(3)

        else:
            stoi_number_str = ""
            species_name = name

        stoi_number: float = float(stoi_number_str) if stoi_number_str else 1.0

        return stoi_number, species_name

    @classmethod
    def from_str(cls, string: str, energy_dict: dict) -> "ReactionStep":
        """Initialize a ReactionStep from a string and an energy dict.

        The string should take the following format:
            *A + 2H2O_g -> 2*B

        Notes:
            1. Use " + "(whitespace in BOTH sides) to separate species
            2. Use "->" to separate reactants and products
            3. For species name format refers to the from_str method
                of Species class
            4. Species without stoichiometric numbers would be treated
                as stoichiometric numbers being 1.

        Also an energy dict is needed:
        energy = {
            "A": (-1, 0),      # Note: no "*"
            "H2O_g": (-2, 3),
            "B": (-4, 0),      # Note: no "*"
        }

        Note:
            The "*" indicating an adsorbed species is unnecessary,
                as the energy of free species (not adsorbed) is need.
        """

        # Check string
        if not isinstance(string, str):
            raise TypeError("Expect a string.")

        string_parts = string.split("->")
        if len(string_parts) != 2:
            raise ValueError("Invalid ReactionStep str.")

        # Split entire string into species parts
        react_parts = string_parts[0].split(" + ")
        product_parts = string_parts[1].split(" + ")

        # Convert each species str to Species for reactants
        react_specs = {}
        for part in react_parts:
            # Separate species_str: "2*CO2(-1, 0)" -> (2.0, "*CO2(-1, 0)")
            number, species_name = cls._sepa_stoi_number(part)

            # Recompile species str to include energy
            species_name = (
                f"{species_name}{energy_dict[species_name.lstrip('*')]}"
            )
            react_specs[Species.from_str(species_name)] = number

        # Convert products
        product_specs = {}
        for part in product_parts:
            number, species_name = cls._sepa_stoi_number(part)

            species_name = (
                f"{species_name}{energy_dict[species_name.lstrip('*')]}"
            )
            product_specs[Species.from_str(species_name)] = number

        return ReactionStep(reactants=react_specs, products=product_specs)


class Reaction:
    """Represent a complete Reaction, as a collection of ReactionStep."""

    def __init__(self, reaction_steps: list[ReactionStep]) -> None:
        self.steps = reaction_steps

    def __len__(self) -> int:
        """Number of ReactionSteps."""
        return len(self.steps)

    def __getitem__(self, index):
        return self.steps[index]

    def __setitem__(self, index, value):
        self.steps[index] = value

    @property
    def steps(self) -> list[ReactionStep]:
        """Core property: collection of ReactionSteps."""

        return self._reaction_steps

    @steps.setter
    def steps(self, reaction_steps: list[ReactionStep]):
        if not all(isinstance(i, ReactionStep) for i in reaction_steps):
            raise TypeError("Each step should be ReactionStep.")

        if len(reaction_steps) != len(set(reaction_steps)):
            raise ValueError("Duplicate ReactionStep found.")

        self._reaction_steps = reaction_steps

    @classmethod
    def from_str(cls, string: str, energy_dict: dict) -> "Reaction":
        """Initialize Reaction from a string.

        The string should be formatted such that each ReactionStep
        resides in a single line. For the format requirements
        of ReactionStep, please refer to its from_str method.
        """
        if not isinstance(string, str):
            raise TypeError("Expect a str.")

        reaction_steps = []
        str_parts = string.strip().split("\n")
        for step in str_parts:
            reaction_steps.append(ReactionStep.from_str(step, energy_dict))

        assert reaction_steps
        return Reaction(reaction_steps)
