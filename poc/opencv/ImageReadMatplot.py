import cv2
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv2.imread('sample.png',0)
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
