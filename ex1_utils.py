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
from typing import List
import cv2
import numpy as np
import matplotlib.pyplot as plt
LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 323010835


# Reads an image and returns the image converted as requested (1 -> grayscale, 2 -> RGB)
def imReadAndConvert(filename:str, representation:int)->np.ndarray:
    image = cv2.imread(filename)
    # if grayscale
    if (representation == 1):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # normalize to [0,1]
        norm_image = cv2.normalize(gray, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return norm_image
    # if rgb
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # normalize to [0,1]
        norm_image = cv2.normalize(rgb, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return norm_image

    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """



# Reads an image as RGB or grayscale and displays it
def imDisplay(filename:str, representation:int)->None:
    src = imReadAndConvert(filename, representation)
    cv2.imshow('im',src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """


# Converts RGB to YIQ
def transformRGB2YIQ(imRGB:np.ndarray)->np.ndarray:
    rgb_to_yiq = np.array([[0.299, 0.587, 0.114],
                           [0.59590059, -0.27455667, -0.32134392],
                           [0.21153661, -0.52273617, 0.31119955]])
    # Matrix multiply
    return np.dot(imRGB, rgb_to_yiq.T.copy())
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """


#Converts YIQ to RGB
def transformYIQ2RGB(imYIQ:np.ndarray)->np.ndarray:
    yiq_to_rgb = np.array([[0.299, 0.587, 0.114],
                           [0.59590059, -0.27455667, -0.32134392],
                           [0.21153661, -0.52273617, 0.31119955]])
    matrix = 1/yiq_to_rgb
    # Matrix multiply
    return np.dot(imYIQ, matrix.T.copy())
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """



# Histogram Equalization
# Returns the histograms and the equalized array
def histogramEqualize(imOrig:np.ndarray)->(np.ndarray,np.ndarray,np.ndarray):
    # Normalize the range back to [0,255]
    imOrig = imOrig*255
    image = imOrig.astype(np.uint8)
    # Show the image
    cv2.imshow('im', image)
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    # CumSum and normalize it
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    # Show the histogram with CumSum
    plt.plot(cdf_normalized, color='b')
    plt.hist(image.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()

    # Span the values according to the algorithm
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    # Transform the values to a 'better' image
    img2 = cdf[image]
    cv2.imshow('im', img2)
    hist, bins = np.histogram(img2.flatten(), 256, [0, 256])

    # CumSum of the new image (should be linear)
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    # Show the histogram with CumSum
    plt.plot(cdf_normalized, color='b')
    plt.hist(img2.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()

    img2 = cv2.normalize(img2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return img2, np.histogram(img2.flatten(), 256, [0, 256]), np.histogram(image.flatten(), 256, [0, 256])
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """



def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    pass

