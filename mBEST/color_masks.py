import cv2
import numpy as np


# pink  75 for dark pictures, 150 for light pictures
def detect_color_pink(img, lw=20, hw=150):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_lim = np.array([0, lw, hw])
    upper_lim = np.array([20, 255, 255])
    mask1 = cv2.inRange(hsv, lower_lim, upper_lim)

    lower_lim = np.array([160, lw, hw])
    upper_lim = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_lim, upper_lim)
    mask = mask1 + mask2
    filter_img = cv2.bitwise_and(img, img, mask=mask)
    filter_img[filter_img > 0] = 255
    return filter_img


# green
def detect_color_green(img, lw=20, hw=150):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_lim = np.array([25, lw, hw])
    upper_lim = np.array([50, 255, 255])
    mask = cv2.inRange(hsv, lower_lim, upper_lim)
    filter_img = cv2.bitwise_and(img, img, mask=mask)
    filter_img[filter_img > 0] = 255
    return filter_img


def detect_color_blue(img, lw=20, hw=75):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_lim = np.array([60, lw, hw])
    upper_lim = np.array([110, 255, 255])
    mask = cv2.inRange(hsv, lower_lim, upper_lim)
    filter_img = cv2.bitwise_and(img, img, mask=mask)
    filter_img[filter_img > 0] = 255
    return filter_img


def detect_color_pink_and_green(img, lw=20, hw=150):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_lim = np.array([0, lw, hw])
    upper_lim = np.array([20, 255, 255])
    mask1 = cv2.inRange(hsv, lower_lim, upper_lim)

    lower_lim = np.array([160, lw, hw])
    upper_lim = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_lim, upper_lim)

    lower_lim = np.array([25, lw, hw])
    upper_lim = np.array([50, 255, 255])
    mask3 = cv2.inRange(hsv, lower_lim, upper_lim)

    mask = mask1 + mask2 + mask3
    filter_img = cv2.bitwise_and(img, img, mask=mask)
    filter_img[filter_img > 0] = 255
    return filter_img
