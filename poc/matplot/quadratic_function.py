import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('Qt5Agg')

# Reference
# https://stackoverflow.com/questions/56656777/userwarning-matplotlib-is-currently-using-agg-which-is-a-non-gui-backend-so


def main():
    x1 = np.linspace(-50, 50, 2000)
    y1 = x1 * x1
    fig, ax = plt.subplots()
    ax.plot(x1, y1, '.')
    plt.show()


if __name__ == '__main__':
    main()
