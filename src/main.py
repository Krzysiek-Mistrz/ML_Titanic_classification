from data.dataset import load_and_split
from preprocessing.preprocessor import build_preprocessor
from models.train import train_random_forest, train_logistic_regression
from evaluation.evaluate import (
    print_classification_report,
    plot_confusion_matrix,
    plot_feature_importances,
)
from prediction.predict import predict_user

def main():
    #features & targets
    features = [
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "class",
        "who",
        "adult_male",
        "alone",
    ]
    target = "survived"

    #load data
    X_train, X_test, y_train, y_test = load_and_split(features, target)

    #build preprocessor
    preprocessor, num_feat, cat_feat = build_preprocessor(X_train)

    #train & eval randomforest model
    rf_model = train_random_forest(X_train, y_train, preprocessor)
    y_pred_rf = rf_model.predict(X_test)
    print_classification_report(y_test, y_pred_rf)
    plot_confusion_matrix(y_test, y_pred_rf, title="Random Forest Confusion Matrix")
    plot_feature_importances(rf_model, num_feat, cat_feat)

    #train & eval logistic reg
    lr_model = train_logistic_regression(X_train, y_train, preprocessor)
    y_pred_lr = lr_model.predict(X_test)
    print_classification_report(y_test, y_pred_lr)
    plot_confusion_matrix(y_test, y_pred_lr, title="Logistic Regression Confusion Matrix")

    #single prediction
    user_input = {
        "pclass": 3,
        "sex": "male",
        "age": 29,
        "sibsp": 0,
        "parch": 0,
        "fare": 7.25,
        "class": "Third",
        "who": "man",
        "adult_male": True,
        "alone": True,
    }
    pred = predict_user(lr_model, user_input)
    result = "survived" if pred == 1 else "dead"
    print(f"prediction: {result}")

if __name__ == "__main__":
    main()