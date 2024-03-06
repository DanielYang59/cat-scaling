"""Class for representing a surface reaction.

Unlike the Reaction class from pymatgen, this object is oriented towards
easier recording of stoichiometric number for surface reactions, as such
species names are treated as is.

Naming rules:
- Species with name starting with "*" are treated as adsorbed species,
    for example "*CO2".

- Species with name ending with physical states including "_g", "_l", "_s"
    and "_aq" are treated as free species, for example "H2_g".
    # TODO: need to rethink about this

"""


from typing import Optional


class Species:
    def __init__(
        self,
        name: str,
        adsorbed: bool,
        state: Optional[str] = None,
    ) -> None:
        self.name = name
        self.adsorbed = adsorbed
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


class Reaction:
    def __init__(
        self,
        reactants: dict[Species, float],
        products: dict[Species, float],
    ) -> None:
        pass

