from digit_detector.annotation import SvhnAnnotation
import os
import glob
import cv2
import numpy as np


def isWrong(label):
    return label < 0 or label > 9


def split_lists(iter_iters, *defaults):
    endOfIter = object()
    iter_iters = iter(iter_iters)
    row = next(iter_iters, endOfIter)

    if row == endOfIter:
        if len(defaults):
            return defaults
        else:
            raise Exception('endOfIter')

    lists = [[item] for item in row]
    expectedRowlen = len(lists)
    for row in iter_iters:
        for i, item in enumerate(row):
            lists[i].append(item)
        if (i + 1) != expectedRowlen:
            raise Exception('All rows must have equal length')
    return lists


def showAnnotation(fileName, boxes, labels):
    green = 0, 200, 0
    red = 0, 0, 200

    img = cv2.imread(fileName)
    minWidth = 150
    ratio = 1
    if img.shape[1] < minWidth:
        ratio = minWidth / img.shape[1]
        img = cv2.resize(img, None, None, fx=ratio, fy=ratio)
    img_rect = img.copy()
    img_label = img.copy()

    for (y1, y2, x1, x2), label in zip(boxes, labels):
        pt1 = int(round(x1 * ratio)), int(round(y1 * ratio))
        pt2 = int(round(x2 * ratio)), int(round(y2 * ratio))
        cv2.rectangle(img_rect, pt1, pt2, green, 1)

        boxCenterX, boxCenterY = (x2 + x1) * ratio / 2, (y1 + y2) * ratio / 2
        text = str(label)
        (textW, textH), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, .5, 1)
        textOrd = int(boxCenterX - textW / 2), int(boxCenterY + textH / 2)
        cv2.putText(img_label, text, textOrd, cv2.FONT_HERSHEY_COMPLEX, .5, red, thickness=1)

    cv2.imshow('Image', np.vstack([img, img_rect, img_label]))
    cv2.setWindowTitle('Image', fileName)
    key = cv2.waitKey()
    return key


def main():
    split = 'train'
    dir = os.path.join('../datasets/SVHN', split)

    annotationFile = os.path.join(dir, 'digitStruct.json')
    annotation = SvhnAnnotation(annotationFile)

    imageFiles = glob.glob(os.path.join(dir, '*.png'))
    for imageFile in sorted(imageFiles):
        boxes, labels = annotation.get_boxes_and_labels(imageFile)
        if showAnnotation(imageFile, boxes, labels) == 27:
            break
        # wrong = [(b, l) for (b, l) in zip(boxes, labels) if isWrong(l)]
        # boxesOfWrongLabels, wrongLabels = split_lists(wrong, [], [])
        # if any(wrongLabels):
        #     showAnnotation(imageFile, boxesOfWrongLabels, wrongLabels)


def test_split_lists():
    list2d = [
        [1, 2, 3],
        [11, 22, 33],
        [111, 222, 333],
        [1111, 2222, 3333]
    ]
    l1, l2, l3 = split_lists(list2d)
    assert l1 == [1, 11, 111, 1111]
    assert l2 == [2, 22, 222, 2222]
    assert l3 == [3, 33, 333, 3333]

    l1, l2, l3 = split_lists([], [], [], [])
    assert l1 == []
    assert l2 == []
    assert l3 == []


if __name__ == '__main__':
    main()
    # test_split_lists()
