import cv2

if __name__ == '__main__':
    img = cv2.imread('sample.png', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', img)
    k = cv2.waitKey(3000)  # Time in millis
    if k == 27:  # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'):  # wait for 's' key to save and exit
        cv2.imwrite('/tmp/sample_gray.png', img)
        cv2.destroyAllWindows()
