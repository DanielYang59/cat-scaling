# TODO: add warning for group member overlap

"""Helper class to record descriptors."""

from __future__ import annotations

from typing import Optional


class Descriptors:
    def __init__(self, groups: dict, method: Optional[str] = None) -> None:
        self._groups = groups
        self._method = method

    def __len__(self) -> int:
        """Number of descriptors."""
        return len(self.groups)

    @property
    def groups(self) -> dict[str, Optional[list[str]]]:
        """Property representing groups of adsorbates.

        For example for CO2 to CH4 reduction reaction:

        With "traditional" method:
        groups = {
            "*CO": ["*COOH", "*CHO", "*CH2O"],  # C-centered group
            "*OH": ["*OCH3", "*O"]              # O-Centered group
        }

        With "adaptive" method:
        groups = {
            "*CO": None,
            "*OH": None
        }
        """

        return self._groups

    @groups.setter
    def groups(self, groups: dict[str, Optional[list[str]]]):
        if not isinstance(groups, dict):
            raise TypeError("Expect groups as dict.")

        for key, value in groups.items():
            if not isinstance(key, str):
                raise TypeError("Keys in groups dictionary must be strings.")
            if value is not None and not isinstance(value, list):
                raise TypeError(
                    "Group members must be lists of strings or None."
                )

        self._groups = groups

    @property
    def descriptors(self) -> list[str]:
        """Name of descriptors as a list."""
        return list(self.groups.keys())

    @property
    def method(self) -> Optional[str]:
        """Method used for building Relation.

        Should be either "traditional" or "adaptive".
        """
        return self._method

    @method.setter
    def method(self, method: Optional[str]):
        if method is not None:
            if method.lower() not in {"traditional", "adaptive"}:
                raise ValueError("Invalid method.")

        self._method = method.lower() if method is not None else None
