import cv2
import numpy as np


def detect_pink_green_teal(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Pink
    lower_pink = np.array([0, 20, 75])
    upper_pink = np.array([20, 255, 255])
    mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)

    lower_pink = np.array([160, 20, 75])
    upper_pink = np.array([180, 255, 255])
    mask_pink += cv2.inRange(hsv, lower_pink, upper_pink)

    # Green
    lower_green = np.array([25, 20, 75])
    upper_green = np.array([50, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Teal
    lower_teal = np.array([50, 20, 20])
    upper_teal = np.array([120, 255, 255])
    mask_teal = cv2.inRange(hsv, lower_teal, upper_teal)

    # Combine masks
    mask = mask_pink + mask_green + mask_teal
    mask[mask != 0] = 255

    # Denoise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask
