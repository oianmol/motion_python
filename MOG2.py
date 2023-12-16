import cv2


def create(parser):
    mog2 = cv2.bgsegm.createBackgroundSubtractorCNT()
    detect_shadows = parser.getboolean("DEFAULT", "detect_shadows")
    return mog2
