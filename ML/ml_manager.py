import cv2
from PIL import Image
import os
import numpy as np
from ML.masks_dict import masks
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt


MIN_WIDTH = 40
MIN_HEIGHT = 40


def detect_people(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    for (xA, yA, xB, yB) in boxes:
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                          (0, 255, 0), 2)
    # out.write(frame.astype('uint8'))
    return frame
    
def detect_led(img):
    
    img = cv2.resize(img, (600, 480), interpolation=cv2.INTER_AREA)        # it's also possible to use INTER_CUBIC interpolation here

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    names_of_mask = masks.keys()

    for curr_name in names_of_mask:
        img_cp = img.copy()

        print(f'curr_name: {curr_name}')
        curr_mask = masks[curr_name]
        hsv_blured = cv2.blur(hsv, (5,5))
        mask = cv2.inRange(hsv_blured, curr_mask[0], curr_mask[1])
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=4)
        # cv2.imshow("Eroded_dilated__mask", mask)
        # plt.imshow(mask)

        contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # find the contours of zone of interests
        contours = contours[0] # extract only contours from received array

        if contours: # if there is any contours
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            cv2.drawContours(img_cp, contours, 0, (255,0,255), 3) # draw contours on the frame

            (x, y, w, h) = cv2.boundingRect(contours[0]) # size and postition of rectangle that circumcribed around contours
            if w >= MIN_WIDTH or h >= MIN_HEIGHT:
                cv2.rectangle(img_cp, (x,y), (x+w, y+h), (0,255,0), 2) # draw rectangle on the frame

        # cv2.imshow('contoures_img', img_cp)
        
        # show some imgs on one plot
        fig, axs = plt.subplots(1, 4)

        fig.suptitle(f'Applying {curr_name} mask')

        axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[0].set_title('Image')
        axs[1].imshow(hsv)
        axs[1].set_title('Hsv')
        axs[2].imshow(hsv_blured)
        axs[2].set_title('Blured')
        axs[3].imshow(mask) # mask
        axs[3].set_title('Eroded_dilated')

        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')

        # plt.show()

    # cv2.imshow('Board1', img)

    # cv2.waitKey(0)
