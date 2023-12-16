import cv2


def create(parser):
    mog2 = cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=15,
                                                    useHistory=True,
                                                    maxPixelStability=15 * 60,
                                                    isParallel=True)
    detect_shadows = parser.getboolean("DEFAULT", "detect_shadows")
    return mog2
