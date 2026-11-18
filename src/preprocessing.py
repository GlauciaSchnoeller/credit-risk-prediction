
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_COLUMNS = ["X2", "X3", "X4"]

ORDINAL_COLUMNS = [
    "X6", "X7", "X8",
    "X9", "X10", "X11"
]

NUMERICAL_COLUMNS = [
    "X1", "X5",
    "X12", "X13", "X14",
    "X15", "X16", "X17",
    "X18", "X19", "X20",
    "X21", "X22", "X23"
]


def create_preprocessor():

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_COLUMNS
            ),
            (
                "numerical",
                StandardScaler(),
                NUMERICAL_COLUMNS
            ),
            (
                "ordinal",
                "passthrough",
                ORDINAL_COLUMNS
            )
        ]
    )
