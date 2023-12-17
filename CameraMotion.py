import logging
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from vidgear.gears import VideoGear

import cv2

import MOG2
import RegionOfInterest
from MotionFileProcessor import MotionFileProcessor


def get_fps(stream):
    return stream.get(5)


def get_width(stream):
    return stream.get(3)


def get_height(stream):
    return stream.get(4)


class CameraMotion:
    def __init__(self, camera_conf_name, camera_id, parser):
        self.video_file_output = None
        self.video_writer = None
        self.t = None
        self.stopped = False

        self.video_start_time = None
        self.motion_file_processor = MotionFileProcessor().start()

        self.camera_id = camera_id
        self.area = parser.getint(camera_conf_name, "area")  # Define Min Contour area
        self.blur = parser.getint("DEFAULT", "blur")
        self.post_motion_wait = parser.getint("DEFAULT", "post_motion_wait")
        self.output_motion_video = parser.getboolean("DEFAULT", "output_motion_video")
        self.width = parser.getint(camera_conf_name, "width")
        self.height = parser.getint(camera_conf_name, "height")
        self.fps = parser.getint(camera_conf_name, "fps")
        self.event_path = str(Path.home()) + os.sep + "events" + os.sep
        self.region_of_interest = parser.get(camera_conf_name, "regions")
        self.video_url = parser.get(camera_conf_name, "uri")
        self.detect_shadows = parser.getboolean("DEFAULT", "detect_shadows")
        self.mog2 = MOG2.create(detect_shadows=self.detect_shadows)
        self.log_config(camera_id)
        Path(self.event_path).mkdir(parents=True, exist_ok=True)
        self.regions = RegionOfInterest.prepare(region_of_interest=self.region_of_interest, width=self.width,
                                                height=self.height)
        # self.video_stream = cv2.VideoCapture(self.video_url, cv2.CAP_FFMPEG)
        self.video_stream = VideoGear(source=self.video_url, stabilize=True, resolution=(self.width, self.height),
                                      framerate=self.fps, camera_num=camera_id, logging=True)

        logging.debug(f" For camera {camera_id} with {self.video_url} created")
        time.sleep(1.0)
        logging.debug(f" For camera {camera_id} starting queue processing now.")
        self.end_time = None
        self.detect_time = None

    def log_config(self, camera_id):
        logging.debug(f"Contour area for camera {camera_id} is {self.area}")
        logging.debug(f"Video Props {self.width} * {self.height} @ {self.fps} fps")
        logging.debug(f"Events will be written to {self.event_path}")
        logging.debug(f"region_of_interest for camera {camera_id} {self.region_of_interest}")
        logging.debug(f"video_url for camera {camera_id} is {self.video_url}")

    def start(self):
        self.t = threading.Thread(target=self.process, args=())
        self.t.start()
        return self

    def join(self):
        self.t.join()
        return self

    def process(self):
        self.video_stream.start()
        while not self.stopped:
            original_frame = self.video_stream.read()
            time.sleep(0.200)
            if original_frame is None:
                # logging.debug(f" For camera {self.camera_id} no more frames, waiting for 2 seconds")
                time.sleep(2.0)
            else:
                # logging.debug(f" For camera {self.camera_id} has frames, will process now")
                try:
                    if self.output_motion_video:
                        if self.video_start_time is not None:
                            diff_time = datetime.now() - self.video_start_time
                            if diff_time >= timedelta(minutes=1):
                                self.video_writer.release()
                                self.video_writer = None
                                self.video_start_time = None
                                self.motion_file_processor.take(video_file_path=self.video_file_output,
                                                                blur=self.blur,
                                                                regions=self.regions, area=self.area,
                                                                camera_id=self.camera_id,
                                                                event_path=self.event_path,
                                                                post_motion_wait=self.post_motion_wait,
                                                                mog2=self.mog2)

                        if self.video_writer is None:
                            self.video_start_time = datetime.now()
                            frame_width = int(self.width)
                            frame_height = int(self.height)
                            frame_size = (frame_width, frame_height)
                            fps = int(self.fps)
                            unique_time = self.video_start_time.strftime('%Y%m%dT%H%M%S')
                            dir_path = self.event_path + self.camera_id + os.sep
                            Path(dir_path).mkdir(parents=True, exist_ok=True)
                            self.video_file_output = dir_path + unique_time + ".mp4"
                            fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
                            self.video_writer = cv2.VideoWriter(self.video_file_output, fourcc, fps, frame_size)
                        if self.video_writer is not None:
                            self.video_writer.write(original_frame)
                except Exception as e:
                    logging.error(e)
        print(f"loop stopped for cameraid {self.camera_id}")
        self.video_stream.stop()
        self.video_writer.release()

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
