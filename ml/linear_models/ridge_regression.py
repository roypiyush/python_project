from sklearn import linear_model


def ridge_regression():
    reg = linear_model.Ridge(alpha=1)
    rg = reg.fit([[1], [2], [3]], [3, 5, 7])
    print("{} Coefficient {}".format(rg, reg.coef_))


if __name__ == '__main__':
    ridge_regression()
