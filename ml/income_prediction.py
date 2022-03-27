import os
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

pd.set_option("display.max_columns", None)
pd.set_option('display.width', None)

column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                'class']
columns_num = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
columns_cat = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
               'native-country']
columns_corr_cat = ['marital-status', 'relationship', 'sex']


def load_data(directory, file_name, names=None, header=None, skiprows=0):
    return pd.read_csv(os.path.join(directory, file_name), names=names, header=header, skiprows=skiprows)


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


def evaluate_model(model, X_prepared, Y_prepared, X_test, Y_test):
    print("**************************************")
    print(model)
    # model.fit(X_prepared, Y_prepared)
#     y_predictions_ = model.predict(X_test)
#     mse_ = mean_squared_error(Y_test, y_predictions_)
#     print("mse on training set", mse_)

    y_predictions_ = model.predict(X_test)
    mse_ = mean_squared_error(Y_test, y_predictions_)
    print("mse on testing set", mse_)

    print("neg_mean_squared_error")
    return model


def scoring_using_cross_validation(m, x, y, s='accuracy'):
    print("**** Start of {} ****".format(s))
    scores_ = cross_val_score(m, x, y, scoring="accuracy", cv=10)
    rmse_scores_ = np.sqrt(scores_)
    display_scores(rmse_scores_)
    print("**** End of {} ****".format(s))


def print_scores(Y_test, Y_predictions):
    mse_ = mean_squared_error(Y_test, Y_predictions)
    print("mse ", mse_)
    acc_ = accuracy_score(Y_test, Y_predictions)
    print("acc ", acc_)
    print("****************************************")


if __name__ == '__main__':
    base_directory = sys.argv[1]
    adult_data_df = load_data(base_directory, 'adult.data', names=column_names)
    adult_test_df = load_data(base_directory, 'adult.test', names=column_names, skiprows=1)

    split = ShuffleSplit(test_size=0.20, random_state=42)
    train_index, test_index = list(split.split(adult_data_df[columns_num]))[0]

    # for train_index, test_index in list(split.split(adult_data_df[columns_num])):
    train_set = adult_data_df.loc[train_index]
    test_set = adult_data_df.loc[test_index]
    # print(train_set.shape)
    # print(test_set.shape)
    # print(test_set.shape[0] / train_set.shape[0])

    # adult_test_df = load_data(base_directory, 'adult.test', skiprows=1)
    # label_enc = LabelEncoder()
    # print(label_enc.fit_transform(adult_data_df['sex']))
    # print(np.c_[label_enc.classes_, label_enc.transform(label_enc.classes_)])

    num_pipeline = Pipeline([
        ('std_scaler', StandardScaler()),
    ])

    X1 = train_set[columns_num].join(train_set[columns_cat])
    X2 = train_set[columns_num].join(train_set[columns_corr_cat])

    ct1 = ColumnTransformer([
        ("norm2", Normalizer(norm='l2'), columns_num),
        ("onehot", OneHotEncoder(), columns_cat)
    ])
    ct2 = ColumnTransformer([
        ("norm2", Normalizer(norm='l2'), columns_num),
        ("onehot", OneHotEncoder(), columns_corr_cat)
    ])
    X1_prepared = ct1.fit_transform(X1)
    X2_prepared = ct2.fit_transform(X2)
    Y_prepared = LabelEncoder().fit_transform(train_set['class'].copy())

    X1_test = ct1.fit_transform(test_set[columns_num].join(test_set[columns_cat]))
    X2_test = ct2.fit_transform(test_set[columns_num].join(test_set[columns_corr_cat]))
    Y_test = LabelEncoder().fit_transform(test_set['class'].copy())  # Y1, Y2 both are same

    X_real = ct2.fit_transform(adult_test_df[columns_num].join(adult_test_df[columns_corr_cat]))
    Y_real = LabelEncoder().fit_transform(adult_test_df['class'].copy())

    sgd = SGDClassifier()
    sgd.fit(X2_prepared, Y_prepared)
    Y_predictions = sgd.predict(X2_test)
    print_scores(Y_test, Y_predictions)

    dt = DecisionTreeClassifier()
    dt.fit(X2_prepared, Y_prepared)
    Y_predictions = dt.predict(X2_test)
    print_scores(Y_test, Y_predictions)

    rf = RandomForestClassifier()
    rf.fit(X2_prepared, Y_prepared)
    Y_predictions = rf.predict(X2_test)
    print_scores(Y_test, Y_predictions)

    print("***** Real *****")
    Y_real_predictions = sgd.predict(X_real)
    print_scores(Y_real, Y_real_predictions)

    Y_real_predictions = dt.predict(X_real)
    print_scores(Y_real, Y_real_predictions)

    Y_real_predictions = rf.predict(X_real)
    print_scores(Y_real, Y_real_predictions)


