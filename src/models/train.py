from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pandas as pd
from sklearn.compose import ColumnTransformer
from typing import Tuple
from sklearn.model_selection import GridSearchCV


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
) -> GridSearchCV:
    """
    Train a RandomForestClassifier with GridSearchCV over n_estimators, max_depth, min_samples_split.
    Returns the fitted GridSearchCV object.
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )
    param_grid = {
        "classifier__n_estimators": [50, 100],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        verbose=2,
    )
    grid.fit(X_train, y_train)
    return grid


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
) -> GridSearchCV:
    """
    Train a LogisticRegression with GridSearchCV over penalty and class_weight.
    Returns the fitted GridSearchCV object.
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(random_state=42, solver="liblinear")),
        ]
    )
    param_grid = {
        "classifier__penalty": ["l1", "l2"],
        "classifier__class_weight": [None, "balanced"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        verbose=2,
    )
    grid.fit(X_train, y_train)
    return grid