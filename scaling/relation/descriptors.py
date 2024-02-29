"""Manage descriptors used for scaling relations."""
# TODO: this may be unnecessary

import warnings
from typing import Optional


class DescriptorManager:
    """Manage descriptors used for scaling relations.

    Args:
        descriptors (Optional[list[str]]): List of descriptors to be set.
            Each descriptor must be a string.

    Raises:
        TypeError: If descriptors is not a list of strings.
        TypeError: If any element of descriptors is not a string.
        ValueError: If duplicate descriptors are provided.
    """

    def __init__(self):
        self._descriptors = None

    @property
    def descriptors(self) -> Optional[list[str]]:
        """Descriptors used for scaling relations."""

        return self._descriptors

    @descriptors.setter
    def descriptors(self, descriptors: Optional[list[str]]):
        if descriptors is not None:
            if not isinstance(descriptors, list):
                raise TypeError("Descriptors must be a list of strings.")

            if not all(isinstance(desc, str) for desc in descriptors):
                raise TypeError("Each descriptor must be a string.")

            if len(descriptors) != len(set(descriptors)):
                raise ValueError("Duplicate descriptors are not allowed.")

            if len(descriptors) > 2:
                warnings.warn(f"Got {len(descriptors)} descriptors.")

        self._descriptors = descriptors
