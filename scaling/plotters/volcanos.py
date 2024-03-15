# TODO: add unit tests

# TODO: selectivity (delta of two limiting potential meshes)

"""Volcano plotter for scaling relations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from scaling.relation import DeltaERelation


def VolcanoPlotter2D(
    relation: DeltaERelation,
    x_range: tuple[float, float, int],
    y_range: tuple[float, float, int],
) -> None:
    """Generate 2D limiting potential volcano plot.

    Args:
        relation (DeltaERelation): The relation object containing data.
        x_range (tuple): A tuple containing (start, stop, num) for x-axis.
        y_range (tuple): A tuple containing (start, stop, num) for y-axis.
    """

    # Generate x/y arrays
    x = np.linspace(*x_range)
    y = np.linspace(*y_range)

    # Generate limiting potential mesh
    # TODO: indexes should be used for rate-determining step (RDS)
    limit_potentials, _rds = relation.eval_limit_potential_2D(x, y)

    # Generate limiting potential volcano plot
    plt.figure(figsize=(8, 6))
    plt.imshow(
        limit_potentials,
        extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
        origin="lower",
        aspect="auto",
    )

    plt.colorbar(label="Limiting Potential (eV)")

    plt.xlabel("Descriptor-X")  # TODO: use species name
    plt.ylabel("Descriptor-Y")
    plt.title("Limiting Potential Volcano Plot")
    plt.show()
