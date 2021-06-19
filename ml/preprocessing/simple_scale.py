from sklearn import preprocessing
import numpy as np

x_train = np.array([[1., 2],
                    [2., 4],
                    [3., 6]])


def simple_scaling():
    print("\nSimple Scaling")
    x_scaled = preprocessing.scale(x_train, with_mean=False)
    print("Scaled Data \n", x_scaled)
    print("Mean ", x_scaled.mean())
    print("Mean axis=0 ", x_scaled.mean(axis=0))
    print("Standard Deviation ", x_scaled.std())
    print("Standard Deviation axis=0", x_scaled.std(axis=0))


def standard_scaling():
    print("\nStandard Scaling")
    x_scaled = preprocessing.StandardScaler().fit(x_train)
    print("Mean {} Scale {}".format(x_scaled.mean_, x_scaled.scale_))
    print("Transform training data ", x_scaled.transform(x_train))
    x_test = [[-1., 1.]]
    print("Transform testing data similar to training data ", x_scaled.transform(x_test))


def minmax_scaling():
    print("\nMin Max Scaling")
    min_max_scaler = preprocessing.MinMaxScaler()
    x_train_minmax = min_max_scaler.fit_transform(x_train)
    print(x_train_minmax)


if __name__ == '__main__':
    simple_scaling()
    standard_scaling()
    minmax_scaling()
