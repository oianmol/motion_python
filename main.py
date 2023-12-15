import argparse
import configparser
import logging
import os
import threading
import time
import time as sleeptime
from datetime import datetime, timedelta
from pathlib import Path
import cv2
import MOG2
import RegionOfInterest
import VideoStreamWriter
from RtspVideoStream import RtspVideoStream


def motion_not_detected(event_path, camera_id):
    end_time_formatted = camera_id + "_MotionEnd_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".txt"
    logging.debug("event ended {date_time}".format(date_time=end_time_formatted))
    open(event_path + end_time_formatted, 'w')


def motion_detected(event_path, camera_id):
    start_time_formatted = camera_id + "_MotionStart_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".txt"
    logging.debug("event started {date_time}".format(date_time=start_time_formatted))
    open(event_path + start_time_formatted, 'w')


def execute(num, camera_id):
    camera_conf_name = "camera_" + num
    # # # # # # # # # # # # # # # # # # # #
    area = parser.getint(camera_conf_name, "area")  # Define Min Contour area
    blur = parser.getint("DEFAULT", "blur")
    post_motion_wait = parser.getint("DEFAULT", "post_motion_wait")
    width = parser.getint(camera_conf_name, "width")
    height = parser.getint(camera_conf_name, "height")
    fps = parser.getint(camera_conf_name, "fps")
    event_path = str(Path.home()) + os.sep + "events" + os.sep
    region_of_interest = parser.get(camera_conf_name, "regions")
    video_url = parser.get(camera_conf_name, "uri")

    logging.debug(f"Contour area for camera {camera_id} is {area}")
    logging.debug(f"Video Props {width} * {height} @ {fps} fps")
    logging.debug(f"Events will be written to {event_path}")
    logging.debug(f"region_of_interest for camera {camera_id} {region_of_interest}")
    logging.debug(f"video_url for camera {camera_id} is {video_url}")

    mog2 = MOG2.create(parser)
    Path(event_path).mkdir(parents=True, exist_ok=True)
    # video_writer = VideoStreamWriter.create(parser, event_path, fps, width, height)
    regions = RegionOfInterest.prepare(region_of_interest=region_of_interest, width=width, height=height)

    video_stream = RtspVideoStream(video_url).start()
    logging.debug(f" For camera {camera_id} with {video_url} created")
    time.sleep(1.0)
    logging.debug(f" For camera {camera_id} waiting to capture frames for 1 second")

    end_time = None
    detect_time = None

    while True:
        if not video_stream.more():
            logging.debug(f" For camera {camera_id} no more frames, waiting for 2 seconds")
            time.sleep(2.0)
        else:
            logging.debug(f" For camera {camera_id} has frames, will process now")
            time.sleep(1.0)  # wait for 1 second before reading frames from queue.
            while video_stream.more():
                original_frame = video_stream.read()
                try:
                    frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.GaussianBlur(frame, (int(blur), int(blur)), 0)
                    frame = RegionOfInterest.mask(frame, regions)
                    final_frame = mog2.apply(frame)
                    # Finding contour of moving object
                    if final_frame is not None:
                        contours, _ = cv2.findContours(final_frame.copy(),
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
                            cv2.putText(original_frame,
                                        'Motion Detected' + datetime.now().strftime("%m-%d-%Y_%H:%M:%S"),
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
                except Exception as e:
                    logging.error(e)

    video_stream.stop()


if __name__ == '__main__':
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'  # Use tcp instead of udp if stream is unstable
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
            camera_id = parser.get("camera_" + str(num), "camera_id")
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
