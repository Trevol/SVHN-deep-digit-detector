import numpy as np
import cv2
import random
import colorsys


def random_colors(N, bright=True):
    # To get visually distinct colors, generate them in HSV space then  convert to RGB.
    # opencv hue range is [0,179], saturation range is [0,255], and value range is [0,255]
    brightness = 255 if bright else 178  # 255*0.7
    hsv = np.uint8([[(int(179 * i / N), 255, brightness) for i in range(N)]])

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR, hsv)
    bgr = list(map(lambda c: tuple(c), bgr[0]))
    random.shuffle(bgr)
    return bgr


def main():
    imageFile = '../datasets/SVHN/test/1.png'
    image = cv2.imread(imageFile, cv2.IMREAD_GRAYSCALE)
    mser = cv2.MSER_create(_delta=1)
    regions, bboxes = mser.detectRegions(image)
    for region, box in zip(regions, bboxes):
        x, y, w, h = box
        print(x, y, w, h)
        drawing = image.copy()
        drawing[region[:, 1], region[:, 0]] = 255
        # for x, y in region:
        #     drawing[y, x] = 255
        cv2.imshow('region', drawing)
        if cv2.waitKey() == 27:
            break



main()
