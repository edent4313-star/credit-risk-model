import pandas as pd

from src.visualization import (
    plot_boxplots
)



def test_boxplot_function():

    df = pd.DataFrame({
        "Amount": [1, 2, 3, 100]
    })

    assert plot_boxplots(df) is None