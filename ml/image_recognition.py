import numpy as np
import cv2


def auto_canny(image, sigma=0.10):
    """
    Defining the autocanny function compute median of image thresholds
    :param image: 2D Array
    :param sigma:
    :return:
    """
    v = np.median(image)
    # apply automatic canny edge detection using the computed median
    lower = int(max(0,(1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged


def main():
    # defining the image, grayscale, blurred
    image = cv2.imread('proxy_duckduckgo_com.jpeg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # apply Canny edge detection using a wide threshold, tight
    # threshold, and automatically determined threshold
    wide = cv2.Canny(blurred, 10, 200)
    tight = cv2.Canny(blurred, 225, 250)
    auto = auto_canny(blurred)
    # show the images
    cv2.imshow("Original", image)
    cv2.imshow("Edges-wide", wide)
    cv2.imshow("Edges-tight", tight)
    cv2.imshow("Edges-auto", auto)
    # Save the images to disk
    cv2.imwrite('/tmp/Wide_config.jpg', wide)
    cv2.imwrite('/tmp/Tight_config.jpg', tight)
    cv2.imwrite('/tmp/Autocanny.jpg', auto)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
