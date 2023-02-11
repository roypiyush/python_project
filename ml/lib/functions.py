
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import roc_auc_score


def using_model_reg(model, scoring, cv, X_train, y_train, X_test, y_test):
    print("****************** {} ******************".format(model))
    if scoring is not None:
        scores_ = cross_val_score(model, X_train, y_train, scoring=scoring, cv=cv)
        print("scoring={} cv={}".format(scoring, cv), scores_.mean(), scores_.std())
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    print("accuracy_score=", accuracy_score(y_test, y_predict))
    return model


def using_model_clf(model, X_train, y_train, X_test, y_test):
    print("****************** {} ******************".format(model))
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    print("accuracy_score=", accuracy_score(y_test, y_predict))
    print("precision_score=", precision_score(y_test, y_predict, average='micro'))
    print("recall_score=", recall_score(y_test, y_predict, average='micro'))
    print("f1_score={}, f1_score_micro={}, f1_score_macro={}"
          .format(f1_score(y_test, y_predict, average=None),
                  f1_score(y_test, y_predict, average='micro'),
                  f1_score(y_test, y_predict, average='macro')))
    return model


def get_roc_auc_score(model, X_train, y_train):
    print(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
    print(roc_auc_score(y_train, model.decision_function(X_train)))


def load_data(directory, file_name, names=None, header=None, skiprows=0):
    return pd.read_excel(os.path.join(directory, file_name), names=names, header=header, skiprows=skiprows)

def load_csv_data(directory, file_name, names=None, header=None, skiprows=0, skipinitialspace=True):
    return pd.read_csv(os.path.join(directory, file_name), names=names, header=header, skiprows=skiprows, skipinitialspace=skipinitialspace)

def load_excel_data(base_directory, file_name):
    path = os.path.join(base_directory, file_name)
    return pd.read_excel(path)
    


def confusion_matrix(model, X_train, Y_train):
    from sklearn.metrics import confusion_matrix

    y_scores_ = cross_val_predict(model, X_train, Y_train, cv=3, method="decision_function")
    model.fit(X_train, Y_train)
    y_predict = model.predict(X_train)
    conf_mx = confusion_matrix(Y_train, y_predict)
    plt.matshow(conf_mx, cmap=plt.cm.gray)
#     print("accuracy_score=", accuracy_score(Y_train, Y_predict))
#     print("precision_score=", precision_score(Y_train, Y_predict))
#     print("recall_score=", recall_score(Y_train, Y_predict))
    print(conf_mx)
    row_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx / row_sums

    np.fill_diagonal(norm_conf_mx, 0)
    plt.matshow(norm_conf_mx, cmap=plt.cm.gray)