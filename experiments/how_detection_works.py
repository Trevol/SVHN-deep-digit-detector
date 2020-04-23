import cv2


def runDetection(img):
    pass


def showDetectionResults():
    pass


def main():
    img_file = '../datasets/SVHN/test/11.png'
    img = cv2.imread(img_file)

    cv2.imshow('', img)
    cv2.waitKey()


main()
