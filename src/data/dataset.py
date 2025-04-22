import seaborn as sns
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import List, Tuple


def load_and_split(
    features: List[str],
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load the Titanic dataset and split into training and test sets.
    Returns: X_train, X_test, y_train, y_test
    """
    df = sns.load_dataset("titanic")
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test