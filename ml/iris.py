import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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


pd.set_option("display.max_columns", None)
pd.set_option('display.width', None)


def load_data(csv_path):
    return pd.read_csv(csv_path)


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


def evaluate_model(model):
    print("**************************************")
    print(model)
    model.fit(X_iris_prepared, Y_iris_prepared)
    predictions_ = model.predict(X_data_train)
    mse_ = mean_squared_error(Y_data_train, predictions_)
    print("mse on training set", mse_)

    predictions_ = model.predict(X_data_test)
    mse_ = mean_squared_error(Y_data_test, predictions_)
    print("mse on testing set", mse_)

    print("neg_mean_squared_error")
    scores_ = cross_val_score(model, X_iris_prepared, Y_iris_prepared, scoring="neg_mean_squared_error", cv=10)
    rmse_scores_ = np.sqrt(-scores_)
    display_scores(rmse_scores_)

    print("accuracy")
    scores_ = cross_val_score(model, X_iris_prepared, Y_iris_prepared, scoring="accuracy", cv=10)
    rmse_scores_ = np.sqrt(scores_)
    display_scores(rmse_scores_)


if __name__ == '__main__':
    cmd_args = sys.argv
    iris_df = load_data(cmd_args[1])
    print(iris_df.describe())
    print(iris_df.info())
    corr_matrix = iris_df.corr()
    print("Correlation Matrix: \n", corr_matrix)

    num_pipeline = Pipeline([
        ('stdscaler', StandardScaler())
    ])

    num_col_trans = ColumnTransformer([
        ('scaler', num_pipeline, ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    ], remainder='passthrough')

    X_iris_prepared = num_col_trans.fit_transform(iris_df.drop("label", axis=1))
    Y_iris_prepared = LabelEncoder().fit_transform(iris_df["label"])

    train_set, test_set = train_test_split(iris_df, test_size=0.2, random_state=42)

    X_data_train = train_set.drop("label", axis=1).to_numpy()
    Y_data_train = LabelEncoder().fit_transform(train_set["label"])

    X_data_test = test_set.drop("label", axis=1).to_numpy()
    Y_data_test = LabelEncoder().fit_transform(test_set["label"])

    evaluate_model(SVC())
    # evaluate_model(DecisionTreeClassifier())
    # evaluate_model(SGDClassifier())
    # evaluate_model(RandomForestClassifier())
