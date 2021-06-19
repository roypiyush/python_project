import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


N = 10
x = np.random.randint(1, 10, N)
y = np.random.randint(1, 10, N)

target = []
for i in range(N):
    hypotenuse = np.hypot([x[i]], [y[i]])
    if hypotenuse < 7:
        plt.plot(x[i], y[i], 'bo')
        target = np.append(target, 0)
    else:
        plt.plot(x[i], y[i], 'ro')
        target = np.append(target, 1)

x_predict, y_predict = np.random.randint(1, 10, 1)[0], np.random.randint(1, 10, 1)[0]


train_coordinates = np.column_stack((x.ravel(), y.ravel()))

clf = LinearDiscriminantAnalysis()
clf.fit(train_coordinates, target.astype('int'))
result = clf.predict([[x_predict, y_predict]]).data[0]
if result == 0:
    plt.plot(x_predict, y_predict, 'bo')
else:
    plt.plot(x_predict, y_predict, 'ro')
plt.annotate('prediction', (x_predict + .1, y_predict + .1))
print("Coefficients {}".format(clf.coef_))
plt.show()
