import cv2


def create(detect_shadows):
    mog2 = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    if bool(detect_shadows):
        mog2.setShadowValue(0)
    return mog2
