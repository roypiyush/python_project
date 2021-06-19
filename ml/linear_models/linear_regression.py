import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


def simple_linear_equation():
    linear_regression = linear_model.LinearRegression()
    # y = 2x + 1
    lg = linear_regression.fit([[1], [2], [3]], [3, 5, 7])
    print("Linear regression {} {}".format(lg, linear_regression.coef_))


def diabetes_example():
    # Load the diabetes data set
    diabetes = datasets.load_diabetes()

    print("Matrix {}".format(diabetes.data.shape))
    # Use only one feature
    diabetes_x = diabetes.data[:, np.newaxis, 2]

    # Split the data into training/testing sets
    diabetes_x_train = diabetes_x[:-20]
    diabetes_x_test = diabetes_x[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]

    # Create linear regression object
    regression = linear_model.LinearRegression()

    # Train the model using the training sets
    regression.fit(diabetes_x_train, diabetes_y_train)

    # Make predictions using the testing set
    diabetes_y_predict = regression.predict(diabetes_x_test)

    # The coefficients
    print('Coefficients: \n', regression.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(diabetes_y_test, diabetes_y_predict))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_predict))

    # Plot outputs
    plt.scatter(diabetes_x_test, diabetes_y_test, color='black')
    plt.plot(diabetes_x_test, diabetes_y_predict, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()


if __name__ == '__main__':
    simple_linear_equation()
    diabetes_example()
