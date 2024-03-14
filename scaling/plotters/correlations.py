# TODO: need polish cosmetically
# TODO: show upper triangle only

"""Plot correlation matrix."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from scaling.data import Eads


def plot_correlation_matrix(
    eads: Eads,
    method: Literal["pearson", "kendall", "spearman"] = "kendall",
    cmap: str = "coolwarm",
) -> None:
    """Plot correlation matrix.

    Parameters:
        eads (Eads): An Eads class containing the data.
        method (Literal["pearson", "kendall", "spearman"], optional): The
            method used to compute the correlation. Defaults to "kendall".
        cmap (str, optional): The colormap to be used. Defaults to "coolwarm".
    """

    # Calculate correlation matrix
    data = eads.data
    corr_matrix = data.corr(method=method)

    plt.matshow(corr_matrix, cmap=cmap)

    # Annotate correlation values
    for i, j in itertools.product(range(len(data.columns)), repeat=2):
        plt.text(
            j,
            i,
            f"{corr_matrix.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            color="white",
        )

    # Set x/y-axis tick labels
    plt.xticks(range(len(data.columns)), data.columns)
    plt.yticks(range(len(data.columns)), data.columns)

    cbar = plt.colorbar()
    cbar.set_label(f"{method.capitalize()} Correlation", fontsize=16)

    plt.show()
