import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import tensorflow as tf
from tensorflow import keras


pd.set_option("display.max_columns", None)
pd.set_option('display.width', None)


def load_data(csv_path):
    return pd.read_csv(csv_path)


def display_scores(prefix_, scores):
    print(prefix_, "Scores:", scores)
    print(prefix_, "Mean:", scores.mean())
    print(prefix_, "Standard deviation:", scores.std())


def evaluate_model(model):
    print("*****************{}*****************".format(model))
    model.fit(X_train, Y_train)
    predictions_ = model.predict(X_train)
    mse_ = mean_squared_error(Y_train, predictions_)
    print("mse on training set", mse_)

    predictions_ = model.predict(X_test)
    mse_ = mean_squared_error(Y_test, predictions_)
    print("mse on testing set", mse_)

    scores_ = cross_val_score(model, X_train, Y_train, scoring="neg_mean_squared_error", cv=5)
    rmse_scores_ = np.sqrt(-scores_)
    display_scores("neg_mean_squared_error", rmse_scores_)

    scores_ = cross_val_score(model, X_train, Y_train, scoring="accuracy", cv=5)
    rmse_scores_ = np.sqrt(scores_)
    display_scores("accuracy", rmse_scores_)


if __name__ == '__main__':
    cmd_args = sys.argv
    iris_df = load_data(os.path.join(cmd_args[1], 'iris.data'))
    numerical_features = iris_df.select_dtypes(include='number').columns.tolist()
    categorical_features = iris_df.select_dtypes(include='object').columns.tolist()

    num_pipeline = Pipeline([
        ('stdscaler', StandardScaler())
    ])
    encoder = LabelEncoder()

    train_set, test_set = train_test_split(iris_df, test_size=0.2, random_state=42)

    k_best = SelectKBest(chi2, k=2)
    k_best.fit(train_set.drop('label', axis=1), train_set["label"])

    full_transformer = ColumnTransformer([
        ('scaler', num_pipeline, k_best.get_feature_names_out().tolist())
    ])
    X_train = full_transformer.fit_transform(train_set)
    Y_train = encoder.fit_transform(train_set["label"])

    X_test = full_transformer.fit_transform(test_set)
    Y_test = encoder.transform(test_set["label"])

    # ada_boost = AdaBoostClassifier(
    #     DecisionTreeClassifier(max_depth=4), n_estimators=500,
    #     algorithm="SAMME.R", learning_rate=0.01)
    #
    # evaluate_model(ada_boost)
    # evaluate_model(LogisticRegression())
    # evaluate_model(KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
    #                                     metric_params=None, n_jobs=1, n_neighbors=3, p=2,
    #                                     weights='uniform'))

    # lg_clf_pipeline = Pipeline([
    #     ('lg', )
    # ])
    grid_search_cv = GridSearchCV(LogisticRegression(multi_class="multinomial"),
                                  param_grid=dict(C=[9, 10, 11], solver=["lbfgs"]),
                                  cv=5,
                                  scoring='neg_mean_squared_error',
                                  return_train_score=True)

    grid_search_cv.fit(X_train, Y_train)

    print("Best cross-validation accuracy: {:.2f}".format(grid_search_cv.best_score_))
    print("Test set score: {:.2f}".format(grid_search_cv.score(X_test, Y_test)))
    print("Best parameters: {}".format(grid_search_cv.best_params_))
    print("Best estimator: {}".format(grid_search_cv.best_estimator_))

    bezdek_iris_df = load_data(os.path.join(cmd_args[1], 'bezdekIris.data'))
    x_bez = full_transformer.fit_transform(bezdek_iris_df)
    y_bez = encoder.transform(bezdek_iris_df['label'])
    y_bez_pred = grid_search_cv.best_estimator_.predict(x_bez)

    print("Accuracy on new data set", accuracy_score(y_bez, y_bez_pred))

    voting_clf = VotingClassifier(
        estimators=[('lr', LogisticRegression()),
                    ('rf', RandomForestClassifier()),
                    ('dt', DecisionTreeClassifier()),
                    ('svc', SVC())],
        voting='hard')
    evaluate_model(voting_clf)
    print("Accuracy on new data set", accuracy_score(y_bez, voting_clf.predict(x_bez)))

    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(), n_estimators=500,
        max_samples=100, bootstrap=True, n_jobs=-1)
    bag_clf.fit(X_train, Y_train)
    print("Accuracy on new data set {}".format(bag_clf), accuracy_score(y_bez, bag_clf.predict(x_bez)))
    print("Accuracy on new data set {}".format(bag_clf), accuracy_score(Y_test, bag_clf.predict(X_test)))

    ada_clf = AdaBoostClassifier(DecisionTreeClassifier(),
        n_estimators=1000, algorithm="SAMME.R", learning_rate=0.5)
    ada_clf.fit(X_train, Y_train)
    print("Accuracy on new data set {}".format(ada_clf), accuracy_score(y_bez, ada_clf.predict(x_bez)))
    print("Accuracy on new data set {}".format(ada_clf), accuracy_score(Y_test, ada_clf.predict(X_test)))

    gdboost_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    gdboost_clf.fit(X_train, Y_train)
    print("Accuracy on new data set {}".format(gdboost_clf), accuracy_score(y_bez, gdboost_clf.predict(x_bez)))
    print("Accuracy on new data set {}".format(gdboost_clf), accuracy_score(Y_test, gdboost_clf.predict(X_test)))