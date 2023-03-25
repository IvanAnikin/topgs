import cv2
from PIL import Image
import os
import numpy as np

def launch():
    print(f'Path: {os.getcwd()}')
    img = cv2.imread(f'{os.getcwd()}/static/images/1.webp')

    img = cv2.resize(img, (600, 480), interpolation=cv2.INTER_AREA)        # it's also possible to use INTER_CUBIC interpolation here

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # green_mask = [np.array([115, 55, 73]), np.array([125, 100, 31])]
    green_mask = [np.array([68, 124, 170]), np.array([99, 255, 255])]

    set_of_masks = [green_mask]

    for curr_mask in set_of_masks:
        cv2.imshow("HSV img", hsv)

        hsv_blured = cv2.blur(hsv, (5,5))
        cv2.imshow("HSV blured img", hsv_blured)

        mask = cv2.inRange(hsv_blured, curr_mask[0], curr_mask[1])
        cv2.imshow("Green_mask", mask)

        mask = cv2.erode(mask, None, iterations=2)
        # cv2.imshow("Eroded_mask", mask)

        mask = cv2.dilate(mask, None, iterations=4)
        cv2.imshow("Eroded_dilated__mask", mask)

    cv2.imshow('Board1', img)

    cv2.waitKey(0)


if __name__ == "__main__":
    launch()