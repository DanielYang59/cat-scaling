"""Helper class to record descriptors."""

from __future__ import annotations

import warnings
from typing import Optional


class Descriptors:
    """Helper class to record descriptors.

    Attributes:
        groups (dict): A dictionary representing groups of adsorbates
            and their respective descriptors.
            Keys are descriptors and values are lists of group members.
            If a group has no members, its value is None.
        method (str, optional): The method used for building Relation.
            Should be either "traditional" or "adaptive".
    """

    def __init__(self, groups: dict, method: Optional[str] = None) -> None:
        self.groups = groups
        self.method = method

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
        """Property:groups setter.

        Warnings:
            A warning would be raised if group members overlap.
        """

        if not isinstance(groups, dict):
            raise TypeError("Expect groups as dict.")

        all_members = []  # for check group member overlap

        for descriptor, members in groups.items():
            if not isinstance(descriptor, str):
                raise TypeError("Keys in groups dictionary must be strings.")
            if members is not None:
                if not isinstance(members, list):
                    raise TypeError(
                        "Group members must be lists of strings or None."
                    )

                all_members.extend(members)

        # Check for group member overlap
        if len(all_members) != len(set(all_members)):
            warnings.warn("Descriptor group members overlap.")

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
        if method is not None and method.lower() not in {
            "traditional",
            "adaptive",
        }:
            raise ValueError("Invalid method.")

        self._method = method.lower() if method is not None else None
