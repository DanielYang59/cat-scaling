# TODO: really need "state" for "species"?


"""Represent a species for a surface reaction."""

import warnings
from math import isclose
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

    def __str__(self) -> str:
        """See from_str for detailed format."""
        prefix = "*" if self.adsorbed else ""

        return (
            f"{prefix}{self.name}_{self.state}"
            f"({self.energy}, {self.correction})"
        )

    def __eq__(self, other: Any) -> bool:
        """The equality comparison."""
        if not isinstance(other, Species):
            return False

        return (
            self.name == other.name
            and self.adsorbed == other.adsorbed
            and isclose(self.energy, other.energy, abs_tol=1e-4)
            and isclose(self.correction, other.correction, abs_tol=1e-4)
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

    @classmethod
    def from_str(cls, string: str) -> "Species":
        """Initialize Species from a string.

        Expect format in:
            *SpeciesName_state(energy, correction)
            where "*"(optional) denotes an adsorbed species,
            "SpeciesName" for the name,
            "_state"(optional) for the physical state,
            "energy" for electronic energy and
            "correction" for any other energy correction terms.

        Some examples:
            "*CO2(-1.0, -2.0)" -> Species(
                "CO2", adsorbed=True, state="NA",
                energy=-1.0, correction=-2.0
                )

            "H2O_g(-2.0, -3.0)" -> Species(
                "H2O", adsorbed=False, state="g",
                energy=-2.0, correction=-3.0
                )
        """
        if not isinstance(string, str):
            raise TypeError("Expect type str.")

        string = string.strip()

        # Check if adsorbed
        adsorbed: bool = True if string.startswith("*") else False

        # Get energy and correction
        e_start = string.find("(")
        e_end = string.find(")")
        energy_parts = string[e_start + 1 : e_end].split(",")
        if e_start == -1 or e_end == -1 or len(energy_parts) != 2:
            raise ValueError("Invalid format for energy and correction.")

        energy = float(energy_parts[0])
        correction = float(energy_parts[1])

        # Get physical state (if available)
        state_start = string.find("_")
        if state_start != -1:
            state: str = string[state_start + 1 : e_start]
        else:
            state = "NA"

        # Get species name
        name = string[:e_start].split("_")[0].lstrip("*")

        return Species(name, energy, adsorbed, correction, state)

    @classmethod
    def from_dict(cls, dct: dict) -> "Species":
        """Initialize Species from a dict."""
        if not isinstance(dct, dict):
            raise TypeError("Expect a dict.")

        name = dct.get("name")
        energy = dct.get("energy")
        adsorbed = dct.get("adsorbed")

        if name is None or energy is None or adsorbed is None:
            raise ValueError("Missing required arg in the dict.")

        return Species(
            name=name,
            energy=energy,
            adsorbed=adsorbed,
            correction=dct.get("correction", 0.0),
            state=dct.get("state", "NA"),
        )
