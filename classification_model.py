import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

titanic = sns.load_dataset('titanic')
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'class', 'who', 'adult_male', 'alone']
target = 'survived'
X = titanic[features]
y = titanic[target]

#analyzing dataset
#print(titanic.head())
#print(titanic.count())
#print(y.value_counts())

#data preparing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
numerical_features = X_train.select_dtypes(include = ['number']).columns.tolist()
categorical_features = X_train.select_dtypes(include = ['object', 'category']).columns.tolist()

#preprocessing pipelines
numerical_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy = 'median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
])
preprocessor = ColumnTransformer(transformers = [
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

#model random forest
pipeline = Pipeline(steps = [
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state = 42))
])
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}
cv = StratifiedKFold(n_splits = 5, shuffle = True)
model = GridSearchCV(
    estimator = pipeline,
    param_grid = param_grid,
    cv = cv,
    scoring = 'accuracy',
    verbose = 2
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#scoring
print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(conf_matrix, annot = True, cmap = 'Blues', fmt = 'd')
plt.title('confusion matrix in titanic classification random forest')
plt.xlabel('predicted')
plt.ylabel('real')
plt.tight_layout()
plt.show()

#feature importances
feature_importances = model.best_estimator_['classifier'].feature_importances_
feature_names = numerical_features + list(model.best_estimator_['preprocessor']
                                        .named_transformers_['cat']
                                        .named_steps['onehot']
                                        .get_feature_names_out(categorical_features))
importance_df = pd.DataFrame({'Feature': feature_names,
                              'Importance': feature_importances
                             }).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.gca().invert_yaxis() 
plt.title('most important features in titanic classification')
plt.xlabel('importance score')
plt.show()
test_score = model.score(X_test, y_test)
print(f"\ntest set accuracy: {test_score:.2%}")


#logistic regression model testing
pipeline.set_params(classifier = LogisticRegression(random_state=42))
model.estimator = pipeline
param_grid = {
    'classifier__solver' : ['liblinear'],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__class_weight' : [None, 'balanced']
}
model.param_grid = param_grid
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#score
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.title('confusion matrix in titanic classification linear regression')
plt.xlabel('predicted')
plt.ylabel('real')
plt.tight_layout()
plt.show()


#user pred
user_input = {
    'pclass': 3,
    'sex': 'male',
    'age': 29,
    'sibsp': 0,
    'parch': 0,
    'fare': 7.25,
    'class': 'Third',
    'who': 'man',
    'adult_male': True,
    'alone': True
}
user_df = pd.DataFrame([user_input])
prediction = model.predict(user_df)
result = "przezyl" if prediction[0] == 1 else "nie przezyl"
print("predykcja dla usera:", result)