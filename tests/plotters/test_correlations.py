import pytest

from cat_scaling.data import Eads
from cat_scaling.plotters.correlations import plot_correlation_matrix
from cat_scaling.utils import PROJECT_ROOT


@pytest.mark.skip("Plotter skipped.")
class Test_plot_correlation_matrix:
    def test_plot(self):
        # Import and load test data
        eads = Eads.from_csv(
            PROJECT_ROOT / "tests" / "relation" / "relation_data.csv"
        )

        plot_correlation_matrix(eads)
