
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import create_preprocessor


def create_logistic_regression():

    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ]
    )


def create_decision_tree():

    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "classifier",
                DecisionTreeClassifier(random_state=42)
            )
        ]
    )


def create_random_forest():

    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )
