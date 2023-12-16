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
from CameraMotion import CameraMotion
from RtspVideoStream import RtspVideoStream

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'  # Use tcp instead of udp if stream is unstable

if __name__ == '__main__':
    process_list = []

    try:
        logging.basicConfig(filename=str(Path.home()) + "/motion_smc_output.log",
                            level=logging.DEBUG,
                            format="%(asctime)s %(message)s")

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
                camera_conf_name = "camera_" + str(num)
                camera_id = parser.get(camera_conf_name, "camera_id")
                disabled = parser.has_option(camera_conf_name, "disabled")
                if not disabled:
                    camera_motion = CameraMotion(camera_conf_name, str(camera_id), parser)
                    process_list.append(camera_motion)

            for process in process_list:
                process.start()

            logging.debug(datetime.now())
        else:
            logging.debug("Config file not provided as arg run with python main.py -c ~/path/to/config.ini")
    except KeyboardInterrupt:
        print('Interrupted')
        for process in process_list:
            process.stop()
