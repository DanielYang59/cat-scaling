"""Handle adsorption (free) energies for linear scaling relations."""


import pandas as pd


class Eads:
    def __init__(self, df: pd.DataFrame) -> None:
        # Check and take arg: df
        if isinstance(df, pd.DataFrame):
            try:
                self.df = df.astype(float)
            except ValueError as e:
                raise ValueError(f"Please double-check input data: {e}.")

        else:
            raise TypeError("Expect data as pd.DataFrame type.")

    def get_adsorbates(self) -> list[str]:
        return self.df.columns.values.tolist()


