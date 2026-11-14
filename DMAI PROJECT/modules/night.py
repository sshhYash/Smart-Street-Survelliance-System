"""Night handling: detect IR / low-light frames and enhance them with CLAHE."""
import cv2
import numpy as np

import config as C

_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def scene_mode(frame):
    small = cv2.resize(frame, (160, 90)).astype(np.int16)
    b, g, r = small[..., 0], small[..., 1], small[..., 2]
    # IR footage has (almost) no colour: the three channels are nearly equal.
    # (HSV saturation alone is unreliable on dark, noisy pixels.)
    colour = (np.abs(b - g) + np.abs(g - r)).mean() / (small.mean() + 5)  # relative to brightness
    if colour < C.IR_COLOUR_THRESHOLD:
        return "IR"
    hsv = cv2.cvtColor(small.astype(np.uint8), cv2.COLOR_BGR2HSV)
    if hsv[..., 2].mean() < C.LOW_LIGHT_THRESHOLD:
        return "LOW-LIGHT"
    return "DAY"


def enhance(frame, mode):
    if mode == "IR":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(_clahe.apply(gray), cv2.COLOR_GRAY2BGR)
    if mode == "LOW-LIGHT":
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        return cv2.cvtColor(cv2.merge((_clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
    return frame
