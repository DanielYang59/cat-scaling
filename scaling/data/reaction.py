# TODO: Need better API to init a Reaction (for example in test_analysis.py)

"""Classes for representing a surface reaction.

This object is oriented towards easier recording of stoichiometric number
for surface reactions (Unlike the Reaction class from pymatgen, t), as such
species names are treated as is (for example name "*CO2" is allowed).
"""

import warnings
from typing import Any


class Species:
    """Represent a species for a surface reaction."""

    def __init__(
        self,
        name: str,
        energy: float,
        adsorbed: bool,
        correction: float = 0.0,
        state: str = "NA",
    ) -> None:
        """Initialize a Species object.

        Args:
            name (str): The name of the species.
            energy (float): energy of the species.
            adsorbed (bool): Whether the species is adsorbed on the surface.
            state (Optional[str], optional): The physical state of the species.
                Valid states are {"g", "l", "s", "aq", "NA"}.

        Raises:
            TypeError: If adsorbed is not a boolean.
            ValueError: If provided state is not valid.
        """

        self.name = name
        self.energy = energy
        self.adsorbed = adsorbed
        self.correction = correction
        self.state = state

    def __eq__(self, other: Any) -> bool:
        """The equality comparison."""
        if not isinstance(other, Species):
            return False

        return (
            self.name == other.name
            and self.adsorbed == other.adsorbed
            and self.energy == other.energy
            and self.state == other.state
        )

    def __hash__(self) -> int:
        return hash((self.name, self.adsorbed, self.energy, self.state))

    @property
    def adsorbed(self) -> bool:
        """Whether the species is adsorbed on the surface."""

        return self._adsorbed

    @adsorbed.setter
    def adsorbed(self, adsorbed: bool):
        if not isinstance(adsorbed, bool):
            raise TypeError("Adsorbed should be boolean.")

        self._adsorbed = adsorbed

    @property
    def state(self) -> str:
        """Physical state of the species.
        Valid states are {"g", "l", "s", "aq", "NA"}.
        """

        return self._state

    @state.setter
    def state(self, state: str):
        valid_states = {"g", "l", "s", "aq", "NA"}
        if state not in valid_states:
            raise ValueError(
                f"Invalid physical state, supported: {valid_states}."
            )

        self._state = state

    @property
    def energy(self) -> float:
        """Energy for the species.

        Note: for an adsorbed species, it's expect to use the free-species
        energy for correct scaling Relation calculation.
        """

        return self._energy

    @energy.setter
    def energy(self, energy: float):
        if not isinstance(energy, (float, int)):
            raise TypeError("Energy should be float.")

        if energy >= 0:
            warnings.warn("Non-negative energy found.")

        self._energy = float(energy)

    @property
    def correction(self) -> float:
        """Optional correction terms, for example zero-point energies."""

        return self._correction

    @correction.setter
    def correction(self, correction: float):
        if not isinstance(correction, (float, int)):
            raise TypeError("Correction should be float.")

        self._correction = float(correction)


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
