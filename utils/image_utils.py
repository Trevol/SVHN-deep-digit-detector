import cv2


def fit_image_to_shape(image, dstShape):
    dstH, dstW = dstShape
    imageH, imageW = image.shape[:2]

    scaleH = dstH / imageH
    scaleW = dstW / imageW
    scale = min(scaleH, scaleW)
    if scale >= 1:
        return image
    return cv2.resize(image, None, None, scale, scale)
