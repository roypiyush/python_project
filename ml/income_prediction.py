import os
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

pd.set_option("display.max_columns", None)
pd.set_option('display.width', None)

column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                'class']
columns_num = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
columns_cat = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
columns_corr_cat = ['marital-status', 'relationship', 'sex']


def load_data(directory, file_name, names=None, header=None, skiprows=0):
    return pd.read_csv(os.path.join(directory, file_name), names=names, header=header, skiprows=skiprows,
                       skipinitialspace=True)


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


def scoring_using_cross_validation(m, x, y, s='accuracy'):
    print("************* Start of Cross Validation {} {} *************".format(s, m))
    scores_ = cross_val_score(m, x, y, scoring="accuracy", cv=10)
    rmse_scores_ = np.sqrt(scores_)
    display_scores(rmse_scores_)
    print("************* End of Cross Validation {} {}*************".format(s, m))


def print_scores(m, Y_test, Y_predictions):

    print("************* {} *************".format(m))
    mse_ = mean_squared_error(Y_test, Y_predictions)
    print("mse ", mse_)
    acc_ = accuracy_score(Y_test, Y_predictions)
    print("acc ", acc_)
    precision_ = precision_score(Y_test, Y_predictions)
    print("precision_ ", precision_)
    recall_ = recall_score(Y_test, Y_predictions)
    print("recall_ ", recall_)
    f1_score_ = f1_score(Y_test, Y_predictions)
    print("f1_score_ ", f1_score_)
    print("****************************************")


def evaluate(x, y, x_test, y_test):
    global lg, sgd, dt, rf

    lg = LogisticRegression(random_state=0, max_iter=1000)
    lg.fit(x, y)
    Y_predictions = lg.predict(x_test)
    print_scores(lg, y_test, Y_predictions)
    scoring_using_cross_validation(lg, x, y, 'accuracy')
    scoring_using_cross_validation(lg, x, y, 'neg_mean_squared_error')

    sgd = SGDClassifier()
    sgd.fit(x, y)

    Y_predictions = sgd.predict(x_test)
    print_scores(sgd, y_test, Y_predictions)
    scoring_using_cross_validation(sgd, x, y, 'accuracy')
    scoring_using_cross_validation(sgd, x, y, 'neg_mean_squared_error')

    dt = DecisionTreeClassifier()
    dt.fit(x, y)
    Y_predictions = dt.predict(x_test)
    print_scores(dt, y_test, Y_predictions)
    scoring_using_cross_validation(dt, x, y, 'accuracy')
    scoring_using_cross_validation(dt, x, y, 'neg_mean_squared_error')

    rf = RandomForestClassifier()
    rf.fit(x, y)
    Y_predictions = rf.predict(x_test)
    print_scores(rf, y_test, Y_predictions)
    scoring_using_cross_validation(rf, x, y, 'accuracy')
    scoring_using_cross_validation(rf, x, y, 'neg_mean_squared_error')


def replace_values(df, colname, value, replace_with):
    df[colname] = df[colname].replace([value], replace_with)


if __name__ == '__main__':
    base_directory = sys.argv[1]
    adult_data_df = load_data(base_directory, 'adult.data', names=column_names)
    replace_values(adult_data_df, 'workclass', '?', 'unknown_class')
    replace_values(adult_data_df, 'occupation', '?', 'unknown_occupation')
    replace_values(adult_data_df, 'native-country', '?', 'unknown_country')
    adult_test_df = load_data(base_directory, 'adult.test', names=column_names, skiprows=1)
    replace_values(adult_test_df, 'workclass', '?', 'unknown_class')
    replace_values(adult_test_df, 'occupation', '?', 'unknown_occupation')
    replace_values(adult_test_df, 'native-country', '?', 'unknown_country')


    # train_set = adult_data_df
    # test_set = adult_test_df

    split = ShuffleSplit(n_splits=20, test_size=0.20, random_state=1)
    train_index, test_index = list(split.split(adult_data_df[columns_num]))[0]

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
        ("norm2", num_pipeline, columns_num),
        ("onehot", OneHotEncoder(), columns_cat)
    ])
    ct2 = ColumnTransformer([
        ("norm2", num_pipeline, columns_num),
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

    evaluate(X2_prepared, Y_prepared, X2_test, Y_test)

    print("***** Real *****")
    Y_real_predictions = sgd.predict(X_real)
    print_scores(lg, Y_real, Y_real_predictions)

    Y_real_predictions = sgd.predict(X_real)
    print_scores(sgd, Y_real, Y_real_predictions)

    Y_real_predictions = dt.predict(X_real)
    print_scores(dt, Y_real, Y_real_predictions)

    Y_real_predictions = rf.predict(X_real)
    print_scores(rf, Y_real, Y_real_predictions)

    # evaluate(X1_prepared, Y_prepared, X1_test, Y_test)
