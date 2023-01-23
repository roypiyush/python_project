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
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

pd.set_option("display.max_columns", None)
pd.set_option('display.width', None)

column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                'class']
columns_num = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
columns_cat = ['marital-status', 'occupation', 'relationship']
columns_corr_cat = ['marital-status', 'relationship', 'sex']


class ColumnDropperTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, columns):
        self.columns = columns

    def transform(self, X, y=None):
        return X.drop(self.columns, axis=1)

    def fit(self, X, y=None):
        return self


class ColumnReplacerTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, column, src_texts, rep_with_text):
        self.column = column
        self.src_texts = src_texts
        self.rep_with_text = rep_with_text

    def transform(self, X, y=None):
        X[self.column].replace(self.src_texts, self.rep_with_text, inplace=True)
        return X

    def fit(self, X, y=None):
        return self


class ColumnLabelEncoderTransformer(TransformerMixin, BaseEstimator):

    def transform(self, X, y=None):
        return X.apply(LabelEncoder().fit_transform)

    def fit(self, X, y=None):
        return self


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
    y_predictions = lg.predict(x_test)
    print_scores(lg, y_test, y_predictions)
    scoring_using_cross_validation(lg, x, y, 'accuracy')
    scoring_using_cross_validation(lg, x, y, 'neg_mean_squared_error')

    sgd = SGDClassifier()
    sgd.fit(x, y)

    y_predictions = sgd.predict(x_test)
    print_scores(sgd, y_test, y_predictions)
    scoring_using_cross_validation(sgd, x, y, 'accuracy')
    scoring_using_cross_validation(sgd, x, y, 'neg_mean_squared_error')

    dt = DecisionTreeClassifier()
    dt.fit(x, y)
    y_predictions = dt.predict(x_test)
    print_scores(dt, y_test, y_predictions)
    scoring_using_cross_validation(dt, x, y, 'accuracy')
    scoring_using_cross_validation(dt, x, y, 'neg_mean_squared_error')

    rf = RandomForestClassifier()
    rf.fit(x, y)
    y_predictions = rf.predict(x_test)
    print_scores(rf, y_test, y_predictions)
    scoring_using_cross_validation(rf, x, y, 'accuracy')
    scoring_using_cross_validation(rf, x, y, 'neg_mean_squared_error')


def replace_values(df, colname, value, replace_with):
    df[colname] = df[colname].replace([value], replace_with)


def using_model(model, scoring, cv):
    print("****************** {} ******************".format(model))
    if scoring is not None:
        scores_ = cross_val_score(model, X_train.toarray(), y_train, scoring=scoring, cv=cv)
        print("scoring={} cv={}".format(scoring, cv), scores_.mean(), scores_.std())
    model.fit(X_train.toarray(), y_train)
    y_predict = model.predict(X_test.toarray())
    print("accuracy_score=", accuracy_score(y_test, y_predict))
    print("precision_score=", precision_score(y_test, y_predict))
    print("recall_score=", recall_score(y_test, y_predict))


if __name__ == '__main__':
    base_directory = sys.argv[1]
    train = load_data(base_directory, 'adult.data', names=column_names)
    test = load_data(base_directory, 'adult.test', names=column_names, skiprows=1)

    #X_train = train.iloc[:, 0:-1]
    #Y_train = train.iloc[:, -1:]

    X_train = train.drop(labels='class', axis=1)
    y_train = train['class']

    preprocess_pipeline = Pipeline([
        ('dropper', ColumnDropperTransformer(['native-country', 'education', 'workclass', 'race', 'sex'])),
        ('unknown_value_replacer', ColumnReplacerTransformer('occupation', ['Other-services'], 'Other-service'))
    ])

    X_train = preprocess_pipeline.fit_transform(X_train)

    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
        #steps=[("scaler", MinMaxScaler())]
    )
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    X_transformer = ColumnTransformer(
        transformers=[
            ("num", Pipeline(
                steps=[("scaler", StandardScaler())]
                # steps=[("scaler", MinMaxScaler())]
            ), columns_num),
            ("cat", categorical_transformer, columns_cat)
        ]
    )

    X_train = X_transformer.fit_transform(X_train)
    y_train = LabelEncoder().fit_transform(y_train)

    # Ready for model

    X_test = test.drop(labels='class', axis=1)
    y_test = test['class'].apply(lambda x: x.replace('.', ''))

    X_test = preprocess_pipeline.fit_transform(X_test)
    X_test = X_transformer.transform(X_test)
    y_test = LabelEncoder().fit_transform(y_test)

    using_model(AdaBoostClassifier(), "accuracy", 10)
    using_model(HistGradientBoostingClassifier(), "accuracy", 10)

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
