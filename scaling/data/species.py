"""Represent a species for a surface reaction."""

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
