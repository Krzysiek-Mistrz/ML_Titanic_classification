import pandas as pd
from typing import Dict


def predict_user(model, user_input: Dict) -> int:
    """
    Given a fitted pipeline (with preprocessor) and a dict of feature values,
    return the predicted class (0 or 1).
    """
    user_df = pd.DataFrame([user_input])
    prediction = model.predict(user_df)[0]
    return prediction