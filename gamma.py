"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from ex1_utils import LOAD_GRAY_SCALE
import numpy as np
import cv2 as cv
PRECISION = 1000


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    image = cv.imread(img_path)
    if rep == LOAD_GRAY_SCALE:
        image = cv.cvtColor(image,cv.COLOR_RGB2GRAY)

    def on_trackbar(val):
        gamma = val/PRECISION
        # clip - Given an interval, values outside the interval are clipped to the interval edges.
        new_image = np.power(image,gamma).clip(0,255).astype(np.uint8)
        cv.imshow('',new_image)

    cv.namedWindow('Gamma')
    cv.createTrackbar('Gamma Track Bar', "Gamma", PRECISION, 2*PRECISION, on_trackbar)
    cv.imshow('', image)
    cv.waitKey()


def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
