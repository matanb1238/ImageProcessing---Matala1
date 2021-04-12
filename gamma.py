import cv2
import nothing as nothing
import numpy as np
import argparse


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


def gammaDisplay(img_path: str, rep: int):
    def nothing(x):
         print(x)
    # img = cv2.imread(img_path, rep)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # gamma = 1.5
    # img1 = np.power(img, gamma)
    # cv2.imshow('img', img1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    alpha_slider_max = 2000
    title_window = 'Gamma Correction'

    # def on_trackbar(val):
    #     alpha = val / alpha_slider_max
    #     beta = (1.0 - alpha)
    #     dst = cv2.addWeighted(image1, alpha, image2, beta, 0.0)
    #     cv2.imshow(title_window, dst)

    image = cv2.imread(img_path, 0)

    cv2.namedWindow(title_window)
    trackbar_name = 'Gamma'
    cv2.createTrackbar(trackbar_name, title_window, 0, int(alpha_slider_max/1000), nothing)
    while(1):
        cv2.imshow('img',image)
        k = cv2.waitKey(1) & 0xFF
        gamma = cv2.getTrackbarPos('Gamma', 'Gamma Correction')
        if(k==27):
            break
        image = np.power(image, gamma)

    cv2.destroyAllWindows()

    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    pass


def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
