"""Classes for representing a surface reaction.

This object is oriented towards easier recording of stoichiometric number
for surface reactions (Unlike the Reaction class from pymatgen, t), as such
species names are treated as is (for example name "*CO2" is allowed).
"""

import warnings
from typing import Optional


class Species:
    """Represent a species for a surface reaction."""

    def __init__(
        self,
        name: str,
        adsorbed: bool,
        energy: Optional[float] = None,
        state: Optional[str] = None,
    ) -> None:
        """Initialize a Species object.

        Args:
            name (str): The name of the species.
            adsorbed (bool): Whether the species is adsorbed on the surface.
            energy (float): energy of the species.
            state (Optional[str], optional): The physical state of the species.
                Valid states are {"g", "l", "s", "aq", "NA"}.

        Raises:
            TypeError: If adsorbed is not a boolean.
            ValueError: If provided state is not valid.
        """

        self.name = name
        self.adsorbed = adsorbed
        self.energy = energy
        self.state = state

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
    def state(self) -> Optional[str]:
        """Physical state of the species.
        Valid states are {"g", "l", "s", "aq", "NA"}.
        """

        return self._state

    @state.setter
    def state(self, state: Optional[str]):
        valid_states = {"g", "l", "s", "aq", "NA"}
        if state is not None and state not in valid_states:
            raise ValueError(
                f"Invalid physical state, supported: {valid_states}."
            )

        self._state = state

    @property
    def energy(self) -> Optional[float]:
        """Energy for the species.

        Note: for an adsorbed species, it's expect to use the free-species
        energy for correct scaling Relation.
        """

        return self._energy

    @energy.setter
    def energy(self, energy: Optional[float]):
        if energy is not None:
            if not isinstance(energy, (float, int)):
                raise TypeError("Energy should be float.")

            if energy > 0:
                warnings.warn("Positive energy found.")

        self._energy = float(energy) if energy is not None else None


class ReactionStep:
    """Represent a single reaction step for surface reaction."""

    def __init__(
        self,
        reactants: dict[Species, float],
        products: dict[Species, float],
    ) -> None:
        """Initialize a Reaction object.

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
    pass
