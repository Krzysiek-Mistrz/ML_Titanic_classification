import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from typing import List


def print_classification_report(y_true, y_pred):
    """Print precision, recall, f1-score."""
    print(classification_report(y_true, y_pred))


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plot a heatmap of the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def plot_feature_importances(
    model: GridSearchCV,
    numerical_features: List[str],
    categorical_features: List[str],
):
    """
    Extract feature importances from the best RandomForest in the pipeline
    and plot them.
    """
    clf = model.best_estimator_["classifier"]
    importances = clf.feature_importances_

    # one‐hot feature names
    ohe = model.best_estimator_["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(ohe.get_feature_names_out(categorical_features))
    feature_names = numerical_features + cat_names

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(df["feature"], df["importance"], color="skyblue")
    plt.gca().invert_yaxis()
    plt.title("Feature Importances")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()