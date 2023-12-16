import cv2


def create(parser):
    mog2 = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    detect_shadows = parser.getboolean("DEFAULT", "detect_shadows")
    if bool(detect_shadows):
        mog2.setShadowValue(0)
    return mog2
