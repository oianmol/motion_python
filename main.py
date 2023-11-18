# Python program to implement  
# Webcam Motion Detector
import argparse
import os
import threading
import configparser
import pandas
# importing datetime class from datetime library
from datetime import datetime, timedelta
import time as sleeptime
# importing OpenCV, time and Pandas library
import cv2
import numpy as np


def execute(num):
    mog = cv2.bgsegm.createBackgroundSubtractorMOG()
    mog2 = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    knn = cv2.createBackgroundSubtractorKNN(detectShadows=False)
    cnt = cv2.bgsegm.createBackgroundSubtractorCNT(isParallel=True)
    # Time of movement
    time = []
    # Initializing DataFrame, one column is start
    # time and other column is end time
    df = pandas.DataFrame(columns=["Start", "End", "Duration"])
    frame1 = None
    frame_no = 0
    motion = 0

    # Constructing a parser
    ap = argparse.ArgumentParser()
    # Adding arguments
    ap.add_argument("-c", "--config", help="Configuration file for MotionSMC")
    args = vars(ap.parse_args())

    parser = configparser.ConfigParser()
    print("Reading config file " + args["config"])
    parser.read(args["config"])

    # Capturing video
    area = parser.getint("basic_config", "area")  # Define Min Contour area
    method = parser.get("basic_config", "method")
    video_url = parser.get("basic_config", "uri")
    event_path = parser.get("basic_config", "event_path")

    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'  # Use tcp instead of udp if stream is unstable
    video = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)
    # get the video frame height and width
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    print("Frame_width: " + str(frame_width))
    print("Frame_height: " + str(frame_height))

    fps = video.get(cv2.CAP_PROP_FPS)
    print("FPS-" + str(fps))

    # convert timestamp into DateTime object
    try:
        original_time = datetime.fromtimestamp(os.path.getmtime(video_url))
    except Exception as e:
        original_time = datetime.now()
        print(e)

    date_time = original_time

    # Infinite while loop to treat stack of image as video
    while video.isOpened():
        # Reading frame(image) from video
        exists, frame = video.read()
        if exists:
            frame_no += 1
            delta = timedelta(milliseconds=int(video.get(cv2.CAP_PROP_POS_MSEC)))
            date_time = original_time + delta
        else:
            if motion == 1:
                time.append(date_time)
            break

        try:
            frame = cv2.GaussianBlur(frame, (7, 7), 0)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Converting color image to gray_scale image
            if method == 'MOG2':
                bgs = mog2.apply(frame)
            if method == 'MOG':
                bgs = mog.apply(frame)
            elif method == 'KNN':
                bgs = knn.apply(frame)
            elif method == 'CNT':
                bgs = cnt.apply(frame)
            elif method == 'ABS':
                # In first iteration we assign the value
                # of static_back to our first frame
                if frame1 is None:
                    frame1 = frame
                    continue
                    # Difference between static background
                    # and current frame(which is GaussianBlur)
                framedelta = cv2.absdiff(frame1, frame)
                # If change in between static background and
                # current frame is greater than 30 it will show white color(255)
                retval, bgs = cv2.threshold(framedelta.copy(), 30, 255, cv2.THRESH_BINARY)
                bgs = cv2.dilate(bgs, None, iterations=2)
        except Exception as e:
            print(e)
            break

        mask = np.zeros_like(frame)

        # Finding contour of moving object
        contours, _ = cv2.findContours(bgs.copy(),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours:
            if cv2.contourArea(contour) < area:
                if motion == 1:
                    time.append(date_time)
                    motion = 0
                continue
            if motion == 0:
                motion = 1
                time.append(date_time)

            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 10), 1)
            cv2.putText(frame, f'{method}', (20, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0, 2))
            cv2.putText(frame, 'Motion Detected', (20, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0, 2))
            cv2.putText(frame, 'date_time ' + date_time.strftime("%m-%d-%Y_%H:%M:%S"), (20, 60),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0, 2))

            cv2.drawContours(mask, contour, -1, 255, 3)
            break

        # cv2.imshow('Original Frame', frame)
        # cv2.imshow(method, bgs)

        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q'):
            # if something is moving then it append the end time of movement
            time.append(date_time)
            print("quit manually")
            break

    video.release()
    # Appending time of motion in DataFrame
    for i in range(0, len(time), 2):
        diff = time[i + 1] - time[i]
        open(event_path + "start_" + time[i].strftime("%m-%d-%Y_%H:%M:%S"),'w')
        open(event_path + "end_" + time[i + 1].strftime("%m-%d-%Y_%H:%M:%S"),'w')
        df = df._append({"Start": time[i], "End": time[i + 1], "Difference": diff}, ignore_index=True)

    # Creating a CSV file in which time of movements will be saved
    df.to_csv(str(num) + method + "_motions.csv")
    # Destroying all the windows
    cv2.destroyAllWindows()


# if __name__ == '__main__':
#    execute()

if __name__ == '__main__':
    process_list = []
    print(datetime.now())

    for num in range(0, 1):
        process = threading.Thread(target=execute, args=([num]))
        process_list.append(process)

    for process in process_list:
        process.start()

    for process in process_list:
        process.join()

    print(datetime.now())
