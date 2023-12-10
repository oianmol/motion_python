# Python program to implement  
# Webcam Motion Detector
import argparse
import configparser
import logging
import os
import threading
import multiprocessing
import time as sleeptime
from collections import namedtuple
# importing datetime class from datetime library
from datetime import datetime, timedelta
from pathlib import Path
import typing as ty

# importing OpenCV, time and Pandas library
import cv2
import numpy as np

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


total_cameras_created = 0
all_instances_created = False


def execute(num, camera_id):
    global total_cameras_created
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'  # Use tcp instead of udp if stream is unstable
    mog2 = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    # # # # # # # # # # # # # # # # # # # #
    end_time = None
    detect_time = None
    area = parser.getint("camera_" + num, "area")  # Define Min Contour area
    logging.debug("Contour area for camera {num} is {area}".format(num=camera_id, area=area))
    blur = parser.defaults().get("blur")
    detect_shadows = parser.getboolean("DEFAULT", "detect_shadows")
    if bool(detect_shadows):
        mog2.setShadowValue(0)
    post_motion_wait = parser.defaults().get("post_motion_wait")
    width = parser.getint("camera_" + num, "width")
    height = parser.getint("camera_" + num, "height")
    fps = parser.getint("camera_" + num, "fps")
    logging.debug("Video Props {width} * {height} @ {fps} fps".format(width=width, height=height, fps=fps))
    event_path = str(Path.home()) + os.sep + "events" + os.sep
    Path(event_path).mkdir(parents=True, exist_ok=True)
    logging.debug("Events will be written to {event_path} ".format(event_path=event_path))

    # # # # # # # # # # # # # # # # # #
    video_writer = None
    video_writer_diff = None
    output_motion_video = parser.getboolean("DEFAULT", "output_motion_video")
    if output_motion_video:
        fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
        logging.debug("video_writer created")
        video_file_output = event_path + 'output.mp4'
        video_file_output_diff = event_path + 'output_diff.mp4'
        logging.debug(video_file_output)
        video_writer = cv2.VideoWriter(video_file_output, fourcc, fps, (width, height), isColor=False)
        video_writer_diff = cv2.VideoWriter(video_file_output_diff, fourcc, fps, (width, height), isColor=False)
    # # # # # # # # # # # # # # # # # # # # #

    # Region of interest start
    region_of_interest = parser.get("camera_" + num, "regions")
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

    video_url = parser.get("camera_" + num, "uri")
    logging.debug("video_url for camera {num} is {video_url}".format(num=camera_id, video_url=video_url))
    video = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)
    total_cameras_created += 1
    logging.debug(
        "video instance created for cameraid {num} with {video_url} total created {total_cameras_created}".format(
            num=camera_id, video_url=video_url, total_cameras_created=total_cameras_created))
    # convert timestamp into DateTime object
    # Infinite while loop to treat stack of image as video
    processing_threads = []
    while True:
        # Reading frame(image) from video
        exists, original_frame = video.read()
        processing_thread = threading.Thread(target=processing_frame,
                                             args=[area, blur, camera_id, detect_time, end_time, event_path, exists,
                                                   mog2, original_frame,
                                                   output_motion_video, post_motion_wait, regions, video_writer,
                                                   video_writer_diff])
        processing_threads.append(processing_thread)
        processing_thread.start()

        for index, thread in enumerate(processing_threads):
            logging.info("Main    : before joining thread %d. cameraid %d", index,camera_id)
            thread.join()
            logging.info("Main    : thread %d done. cameraid %d", index,camera_id)

    logging.error(f"thread finished for camera {camera_id}")
    video.release()
    if output_motion_video:
        logging.debug("releasing video_writer")
        video_writer.release()
        video_writer_diff.release()
    # Destroying all the windows
    cv2.destroyAllWindows()


def processing_frame(area, blur, camera_id, detect_time, end_time, event_path, exists, mog2, original_frame,
                     output_motion_video, post_motion_wait, regions, video_writer, video_writer_diff):
    if exists:
        try:
            frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.GaussianBlur(frame, (int(blur), int(blur)), 0)
            mask = np.zeros_like(frame, dtype=np.uint8)
            for shape in [regions]:
                points = np.array([shape], np.int32)
                mask = cv2.fillPoly(mask, points, color=(255, 255, 255), lineType=cv2.LINE_4)
            frame = np.bitwise_and(frame, mask).astype(np.uint8)
            bgs = mog2.apply(frame)
            # Finding contour of moving object
            if bgs is not None:
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
                    cv2.putText(original_frame, 'Motion Detected' + datetime.now().strftime("%m-%d-%Y_%H:%M:%S"),
                                (20, 300), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255, 2))
                    cv2.drawContours(image=original_frame, contours=contours_filtered, contourIdx=-1, color=255,
                                     thickness=3)

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
                    cv2.putText(original_frame, 'Time Elapsed Post Motion End {time}'.format(time=diff / 1000),
                                (20, 250),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255, 2))
                    if new_end_time < datetime.now():
                        motion_not_detected(event_path, camera_id)
                        end_time = None
                        detect_time = None
                if output_motion_video:
                    video_writer.write(frame)
                    video_writer_diff.write(bgs)
        except Exception as e:
            logging.error(e)
    else:
        sleeptime.sleep(2)
        logging.error(f"no frame discovered for {camera_id} will retry after 2 seconds")


if __name__ == '__main__':
    logging.basicConfig(filename=str(Path.home()) + "/motion_smc_output.log",
                        level=logging.DEBUG,
                        format="%(asctime)s %(message)s")

    process_list = []

    logging.debug("Program started at {time} ".format(time=datetime.now()))
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
            has_option = parser.has_option("camera_" + str(num), "disabled")
            if not has_option:
                process = threading.Thread(target=execute, args=([str(num), str(camera_id)]))
                process_list.append(process)

        for process in process_list:
            process.start()

        for process in process_list:
            process.join()

        logging.debug(datetime.now())
    else:
        logging.debug("Config file not provided as arg run with python main.py -c ~/path/to/config.ini")
