import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier

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


class ColumnDropperTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, columns):
        self.columns = columns

    def transform(self, X, y=None):
        for c in list(X):
            for n in self.columns:
                if c in n:
                    X.drop(c, axis=1, inplace=True)
        return X

    def fit(self, X, y=None):
        return self


class ColumnUnknownValueTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, columns):
        self.columns = columns

    def transform(self, X, y=None):
        for c in self.columns:
            X[c].replace(['?'], 'unknown_{}'.format(c), inplace=True)
        return X

    def fit(self, X, y=None):
        return self


def using_model(model, scoring, cv):
    print("****************** {} ******************".format(model))
    if scoring is not None:
        scores_ = cross_val_score(model, X_train, Y_train, scoring=scoring, cv=cv)
        print("scoring={} cv={}".format(scoring, cv), scores_.mean(), scores_.std())
    model.fit(X_train, Y_train)
    Y_predict = model.predict(X_test)
    print("accuracy_score=", accuracy_score(Y_test, Y_predict))
    print("precision_score=", precision_score(Y_test, Y_predict))
    print("recall_score=", recall_score(Y_test, Y_predict))


if __name__ == '__main__':
    base_directory = sys.argv[1]
    adult_data_df = load_data(base_directory, 'adult.data', names=column_names)
    adult_test_df = load_data(base_directory, 'adult.test', names=column_names, skiprows=1)

    preprocess_pipeline = Pipeline([
        ('dropper', ColumnDropperTransformer(['education'])),
        ('unknown_value_replacer', ColumnUnknownValueTransformer(['workclass', 'occupation', 'native-country']))
    ])
    # adult_data_df = preprocess_pipeline.fit_transform(adult_data_df)
    # adult_test_df = preprocess_pipeline.fit_transform(adult_test_df)

    num_pipeline = Pipeline([
        ('std_scaler', StandardScaler()),
    ])

    one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
    ct = ColumnTransformer(transformers=[
        ("norm2", num_pipeline, columns_num),
        ('onehot', one_hot_encoder, columns_cat)
    ])

    X_train = adult_data_df.drop(labels='class', axis=1)
    Y_train = adult_data_df['class']

    X_test = adult_test_df.drop(labels='class', axis=1)
    Y_test = adult_test_df['class'].apply(lambda x: x.replace('.', ''))

    label_encoder = LabelEncoder()
    label_encoder.fit(Y_train)
    X_train = ct.fit_transform(X_train)
    Y_train = label_encoder.transform(Y_train)

    X_test = ct.transform(X_test)
    Y_test = label_encoder.transform(Y_test.apply(lambda x: x.replace('.', '')))

    using_model(SGDClassifier(), "accuracy", 10)
    using_model(RandomForestClassifier(), "accuracy", 10)

    lg_newton_cg = LogisticRegression(max_iter=500, solver='newton-cg')
    using_model(lg_newton_cg, "neg_mean_squared_error", 10)
    using_model(lg_newton_cg, "accuracy", 10)

    lg_lgbfs = LogisticRegression(max_iter=1000)
    using_model(lg_lgbfs, "neg_mean_squared_error", 10)
    using_model(lg_lgbfs, "accuracy", 10)
    using_model(DecisionTreeClassifier(), "neg_mean_squared_error", 10)
    using_model(DecisionTreeClassifier(), "accuracy", 10)


    # split = ShuffleSplit(n_splits=20, test_size=0.20, random_state=1)
    # train_index, test_index = list(split.split(adult_data_df[columns_num]))[0]
    #
    # X = pd.get_dummies(adult_data_df.drop('class', axis=1))
    # y = adult_data_df['class'].copy()
    # X_train = X.loc[train_index]
    # X_test = X.loc[test_index]
    #
    # label_encoder = LabelEncoder()
    # Y_train = label_encoder.fit_transform(y.loc[train_index])
    # Y_test = label_encoder.transform(y.loc[test_index])
    #
    # X1_train = ct.fit_transform(X_train)
    # X2_train = ct.transform(ColumnDropperTransformer(list(set(columns_cat) - set(columns_corr_cat)))
    #                             .fit_transform(X_train))
    #
    # X1_test = ct.transform(X_test)
    # X2_test = ct.transform(ColumnDropperTransformer(list(set(columns_cat) - set(columns_corr_cat)))
    #                            .fit_transform(X_test))
    #
    # #evaluate(X1_train, Y_train, X1_test, Y_test)
    # evaluate(X2_train, Y_train, X2_test, Y_test)

    # X_real = ct.transform(adult_test_df.drop('class', axis=1))
    # Y_real = LabelEncoder().fit_transform(adult_test_df['class'].copy())

    # print("***** Real *****")
    # Y_real_predictions = sgd.predict(X_real)
    # print_scores(lg, Y_real, Y_real_predictions)
    #
    # Y_real_predictions = sgd.predict(X_real)
    # print_scores(sgd, Y_real, Y_real_predictions)
    #
    # Y_real_predictions = dt.predict(X_real)
    # print_scores(dt, Y_real, Y_real_predictions)
    #
    # Y_real_predictions = rf.predict(X_real)
    # print_scores(rf, Y_real, Y_real_predictions)

    # evaluate(X1_prepared, Y_prepared, X1_test, Y_test)
