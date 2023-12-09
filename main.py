# Python program to implement  
# Webcam Motion Detector
import argparse
import os
import threading
import configparser
from collections import namedtuple
import logging

# importing datetime class from datetime library
from datetime import datetime, timedelta
import time as sleeptime
# importing OpenCV, time and Pandas library
import cv2
import numpy as np
from pathlib import Path
import typing as ty

Point = namedtuple("Point", ['x', 'y'])
Size = namedtuple("Size", ['w', 'h'])


def motion_not_detected(event_path, camera_id):
    end_time_formatted = camera_id + "_MotionEnd_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".txt"
    logging.debug("event ended {date_time}".format(date_time=end_time_formatted))
    open(event_path + end_time_formatted, 'w')


def motion_detected(event_path, camera_id):
    "%t_MotionStart_%Y_%m_%d_%H_%M_%S.txt"
    start_time_formatted = camera_id + "_MotionStart_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".txt"
    logging.debug("event started {date_time}".format(date_time=start_time_formatted))
    open(event_path + start_time_formatted, 'w')


def initial_point_list(w: int, h: int) -> ty.List[Point]:
    # For now start with a rectangle covering 1/4 of the frame in the middle.
    top_left = Point(x=0, y=0)
    box_size = Size(w=w, h=h)
    return [
        top_left,
        Point(x=top_left.x + box_size.w, y=top_left.y),
        Point(x=top_left.x + box_size.w, y=top_left.y + box_size.h),
        Point(x=top_left.x, y=top_left.y + box_size.h),
    ]


def execute(camera_id):
    mog = cv2.bgsegm.createBackgroundSubtractorMOG()
    mog2 = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

    knn = cv2.createBackgroundSubtractorKNN(detectShadows=True)
    cnt = cv2.bgsegm.createBackgroundSubtractorCNT(isParallel=True)

    frame1 = None
    end_time = None
    detect_time = None

    area = parser.getint("camera_" + camera_id, "area")  # Define Min Contour area
    logging.debug("Contour area for camera {num} is {area}".format(num=camera_id, area=area))

    blur = parser.defaults().get("blur")
    detect_shadows = parser.defaults().get("detect_shadows")
    if bool(detect_shadows):
        mog2.setShadowValue(0)

    post_motion_wait = parser.defaults().get("post_motion_wait")
    method = parser.defaults().get("method")
    logging.debug("Method used for camera {num} is {method}".format(num=camera_id, method=method))

    video_url = parser.get("camera_" + camera_id, "uri")
    logging.debug("video_url for camera {num} is {video_url}".format(num=camera_id, video_url=video_url))

    width = parser.getint("camera_" + camera_id, "width")
    height = parser.getint("camera_" + camera_id, "height")
    fps = parser.getint("camera_" + camera_id, "fps")
    logging.debug("Video Props {width} * {height} @ {fps} fps".format(width=width, height=height, fps=fps))

    event_path = parser.get("camera_" + camera_id, "event_path")
    Path(event_path).mkdir(parents=True, exist_ok=True)
    logging.debug("Events will be written to {event_path} ".format(event_path=event_path))

    output_motion_video = parser.defaults().get("output_motion_video")
    if output_motion_video:
        fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
        logging.debug("video_writer created")
        video_file_output = event_path + 'output.mp4'
        video_file_output_diff = event_path + 'output_diff.mp4'
        logging.debug(video_file_output)
        video_writer = cv2.VideoWriter(video_file_output, fourcc, fps, (width, height), isColor=False)
        video_writer_diff = cv2.VideoWriter(video_file_output_diff, fourcc, fps, (width, height), isColor=False)

    # Region of interest start
    region_of_interest = parser.get("camera_" + camera_id, "regions")
    logging.debug("region_of_interest for camera {num} {region_of_interest} ".format(num=camera_id,
                                                                                     region_of_interest=region_of_interest))
    regions = []
    if len(region_of_interest) > 0:
        x = region_of_interest.split(" ")
        it = iter(list(map(int, x)))
        for x in it:
            regions.append(Point(x, next(it)))

    if len(regions) == 0:
        initial_region = [initial_point_list(w=width, h=height)]
        regions = initial_region

    # Region of interest end

    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'  # Use tcp instead of udp if stream is unstable
    video = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)

    # convert timestamp into DateTime object
    # Infinite while loop to treat stack of image as video
    while True:
        # Reading frame(image) from video
        exists, original_frame = video.read()
        sleeptime.sleep(0.250)
        if exists:
            delta = timedelta(milliseconds=int(video.get(cv2.CAP_PROP_POS_MSEC)))
        else:
            logging.error("no frame discovered...")

        try:
            # test_frame = original_frame.copy()
            # test_frame = cv2.normalize(original_frame, test_frame, -255, 512, cv2.NORM_MINMAX)
            frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.GaussianBlur(frame, (int(blur), int(blur)), 0)

            mask = np.zeros_like(frame, dtype=np.uint8)
            for shape in [regions]:
                points = np.array([shape], np.int32)
                mask = cv2.fillPoly(mask, points, color=(255, 255, 255), lineType=cv2.LINE_4)
            # TODO: We can pre-calculate a masked version of the frame and just swap both out.
            frame = np.bitwise_and(frame, mask).astype(np.uint8)

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
                frame1 = None
        except Exception as e:
            logging.error(e)
            break

        # Finding contour of moving object
        contours, _ = cv2.findContours(bgs.copy(),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        has_motion = []
        contours_filtered = []
        for contour in contours:
            if cv2.contourArea(contour) < area:
                has_motion.append(False)
            else:
                contours_filtered.append(contour)
                has_motion.append(True)

        contour_has_motion = any(has_motion)
        if contour_has_motion:
            cv2.putText(original_frame, 'Motion Detected' + datetime.now().strftime("%m-%d-%Y_%H:%M:%S"), (20, 300),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255, 2))
            cv2.drawContours(image=original_frame, contours=contours_filtered, contourIdx=-1, color=255, thickness=3)

        if len(has_motion) > 0:
            if contour_has_motion and detect_time is None:
                detect_time = datetime.now()
                end_time = None
                motion_detected(event_path, camera_id)
            if contour_has_motion and detect_time is not None:
                detect_time = datetime.now()
                end_time = None
            else:
                # we do not have any motion
                if end_time is None and detect_time is not None:
                    end_time = datetime.now()
        else:
            # we do not have any motion
            if end_time is None and detect_time is not None:
                end_time = datetime.now()

        if end_time is not None:
            new_end_time = end_time + timedelta(seconds=int(post_motion_wait))
            diff = new_end_time - datetime.now()
            cv2.putText(original_frame, 'Time Elapsed Post Motion End {time}'.format(time=diff / 1000), (20, 250),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255, 2))
            if new_end_time < datetime.now():
                motion_not_detected(event_path, camera_id)
                end_time = None
                detect_time = None

        # cv2.imshow('bgs', bgs)
        # cv2.imshow('Original Frame', original_frame)
        # cv2.imshow('Frame', frame)

        if output_motion_video:
            video_writer.write(frame)
            video_writer_diff.write(bgs)

        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q'):
            logging.debug("quit manually")
            break

    logging.debug("video released")
    video.release()
    if output_motion_video:
        logging.debug("releasing video_writer")
        video_writer.release()
        video_writer_diff.release()
    # Destroying all the windows
    cv2.destroyAllWindows()


if __name__ == '__ma in__':
    logging.debug(datetime.now())

    # Constructing a parser
    ap = argparse.ArgumentParser()
    # Adding arguments
    ap.add_argument("-c", "--config", help="Configuration file for MotionSMC")
    args = vars(ap.parse_args())

    parser = configparser.ConfigParser()
    logging.debug("Reading config file " + args["config"])
    parser.read(args["config"])

    # Capturing video
    cameras = parser.defaults().get("cameras")
    logging.debug("Total cameras " + str(cameras))
    execute("0")

if __name__ == '__main__':
    logging.basicConfig(filename="motion_smc_output.log",
                        level=logging.DEBUG,
                        format="%(asctime)s %(message)s")

    process_list = []

    logging.debug("Program started at {time}!".format(time=datetime.now()))
    # Constructing a parser
    ap = argparse.ArgumentParser()
    # Adding arguments
    ap.add_argument("-c", "--config", help="Configuration file for MotionSMC")
    args = vars(ap.parse_args())
    config_file = str(args["config"])

    if not os.path.isfile(config_file):
        config_file = str(Path.home()) + "/motion_smc.ini"

    if os.path.isfile(config_file):
        logging.debug("Reading config file " + config_file)
        parser = configparser.ConfigParser()
        parser.read(config_file)
        # Capturing video
        cameras = int(parser.defaults().get("cameras"))
        logging.debug("Running for {total} cameras ".format(total=str(cameras)))
        for num in range(0, cameras):
            camera_id = parser.getint("camera_" + str(num), "camera_id")
            process = threading.Thread(target=execute, args=([str(camera_id)]))
            process_list.append(process)

        for process in process_list:
            process.start()

        for process in process_list:
            process.join()

        logging.debug(datetime.now())
    else:
        logging.debug("Config file not provided as arg run with python main.py -c ~/path/to/config.ini")
