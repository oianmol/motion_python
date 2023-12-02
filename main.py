# Python program to implement  
# Webcam Motion Detector
import argparse
import os
import threading
import configparser
# importing datetime class from datetime library
from datetime import datetime, timedelta
import time as sleeptime
# importing OpenCV, time and Pandas library
import cv2
import numpy as np
from pathlib import Path


def execute(num):
    mog = cv2.bgsegm.createBackgroundSubtractorMOG()
    mog2 = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    knn = cv2.createBackgroundSubtractorKNN(detectShadows=False)
    cnt = cv2.bgsegm.createBackgroundSubtractorCNT(isParallel=True)

    frame1 = None
    frame_no = 0
    motion = 0

    area = parser.getint("camera_" + num, "area")  # Define Min Contour area
    print("Contour area for camera {num} is {area}".format(num=num, area=area))

    method = parser.get("basic_config", "method")
    print("Method used for camera {num} is {method}".format(num=num, method=method))

    video_url = parser.get("camera_" + num, "uri")
    print("video_url for camera {num} is {video_url}".format(num=num, video_url=video_url))

    width = parser.getint("camera_" + num, "width")
    height = parser.getint("camera_" + num, "height")
    fps = parser.getint("camera_" + num, "fps")
    print("Video Props {width} * {height} @ {fps} fps".format(width=width, height=height, fps=fps))

    event_path = parser.get("camera_" + num, "event_path")
    Path(event_path).mkdir(parents=True, exist_ok=True)
    print("Events will be written to {event_path} ".format(event_path=event_path))

    output_motion_video = parser.getboolean("basic_config", "output_motion_video")
    fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
    print("video_writer created")
    video_file_output = event_path + 'output.mp4'
    video_file_output_diff = event_path + 'output_diff.mp4'
    print(video_file_output)
    video_writer = cv2.VideoWriter(video_file_output, fourcc, fps, (width, height), isColor=False)
    video_writer_diff = cv2.VideoWriter(video_file_output_diff, fourcc, fps, (width, height), isColor=False)

    region_of_interest = parser.get("camera_" + num, "region_of_interest")
    print("region_of_interest for camera {num} {region_of_interest} ".format(num=num,
                                                                             region_of_interest=region_of_interest))

    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'  # Use tcp instead of udp if stream is unstable
    video = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)

    # convert timestamp into DateTime object
    # Infinite while loop to treat stack of image as video
    while True:
        # Reading frame(image) from video
        exists, original_frame = video.read()
        date_time = datetime.now()
        # sleeptime.sleep(0.250)
        if exists:
            frame_no += 1
            delta = timedelta(milliseconds=int(video.get(cv2.CAP_PROP_POS_MSEC)))
        else:
            if motion == 1:
                print("event ended {date_time}".format(date_time=date_time.strftime("%m-%d-%Y_%H:%M:%S:%f")))
                open(event_path + "end_" + date_time.strftime("%m-%d-%Y_%H:%M:%S:%f"), 'w')
            break

        try:
            frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.GaussianBlur(frame, (7, 7), 0)
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
                    print("event ended {date_time}".format(date_time=date_time.strftime("%m-%d-%Y_%H:%M:%S:%f")))
                    open(event_path + "end_" + date_time.strftime("%m-%d-%Y_%H:%M:%S:%f"), 'w')
                    motion = 0
                continue
            if motion == 0:
                motion = 1
                print("event started {date_time}".format(date_time=date_time.strftime("%m-%d-%Y_%H:%M:%S:%f")))
                open(event_path + "start_" + date_time.strftime("%m-%d-%Y_%H:%M:%S:%f"), 'w')

            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 10), 1)
            cv2.putText(frame, f'{method}', (20, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255, 2))
            cv2.putText(frame, 'Motion Detected', (20, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255, 2))
            cv2.putText(frame, 'date_time ' + date_time.strftime("%m-%d-%Y_%H:%M:%S"), (20, 60),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255, 2))

            cv2.drawContours(mask, contour, -1, 255, 3)
            break

        cv2.imshow('Original Frame', original_frame)
        cv2.imshow('Frame', frame)
        cv2.imshow(method, bgs)

        if output_motion_video:
            video_writer.write(frame)
            video_writer_diff.write(bgs)

        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q'):
            open(event_path + "end_" + date_time.strftime("%m-%d-%Y_%H:%M:%S:%f"), 'w')
            print("quit manually")
            break

    print("video released")
    video.release()
    if output_motion_video:
        print("releasing video_writer")
        video_writer.release()
        video_writer_diff.release()
    # Destroying all the windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print(datetime.now())

    # Constructing a parser
    ap = argparse.ArgumentParser()
    # Adding arguments
    ap.add_argument("-c", "--config", help="Configuration file for MotionSMC")
    args = vars(ap.parse_args())

    parser = configparser.ConfigParser()
    print("Reading config file " + args["config"])
    parser.read(args["config"])

    # Capturing video
    cameras = parser.getint("basic_config", "cameras")
    print("Total cameras " + str(cameras))
    execute("0")

if __name__ == '__ma in__':
    process_list = []
    print(datetime.now())

    # Constructing a parser
    ap = argparse.ArgumentParser()
    # Adding arguments
    ap.add_argument("-c", "--config", help="Configuration file for MotionSMC")
    args = vars(ap.parse_args())

    parser = configparser.ConfigParser()
    print("Reading config file " + args["config"])
    parser.read(args["config"])

    # Capturing video
    cameras = parser.getint("basic_config", "cameras")
    print("Total cameras " + str(cameras))
    for num in range(0, cameras):
        process = threading.Thread(target=execute, args=([str(num)]))
        process_list.append(process)

    for process in process_list:
        process.start()

    for process in process_list:
        process.join()

    print(datetime.now())
