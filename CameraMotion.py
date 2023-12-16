import logging
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2

import MOG2
import RegionOfInterest
from MotionFileProcessor import MotionFileProcessor
from VideoStreamer import VideoStreamer


class CameraMotion:
    def __init__(self, camera_conf_name, camera_id, parser):
        self.video_file_output = None
        self.video_writer = None
        self.t = None
        self.stopped = False

        self.video_start_time = None

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
        self.log_config(camera_id)
        self.mog2 = MOG2.create(parser)
        Path(self.event_path).mkdir(parents=True, exist_ok=True)
        self.regions = RegionOfInterest.prepare(region_of_interest=self.region_of_interest, width=self.width,
                                                height=self.height)
        self.video_stream = VideoStreamer(self.video_url).start()
        logging.debug(f" For camera {camera_id} with {self.video_url} created")
        time.sleep(1.0)
        logging.debug(f" For camera {camera_id} starting queue processing now.")
        self.end_time = None
        self.detect_time = None
        self.motion_file_processor = MotionFileProcessor(blur=self.blur, regions=self.regions, area=self.area,
                                                         camera_id=camera_id, event_path=self.event_path,
                                                         post_motion_wait=self.post_motion_wait, mog2=self.mog2)

    def log_config(self, camera_id):
        logging.debug(f"Contour area for camera {camera_id} is {self.area}")
        logging.debug(f"Video Props {self.width} * {self.height} @ {self.fps} fps")
        logging.debug(f"Events will be written to {self.event_path}")
        logging.debug(f"region_of_interest for camera {camera_id} {self.region_of_interest}")
        logging.debug(f"video_url for camera {camera_id} is {self.video_url}")

    def start(self):
        self.t = threading.Thread(target=self.process, args=())
        self.t.daemon = True
        self.t.start()
        return self

    def join(self):
        self.t.join()
        return self

    def process(self):
        while not self.stopped:
            if not self.video_stream.more():
                # logging.debug(f" For camera {self.camera_id} no more frames, waiting for 2 seconds")
                time.sleep(2.0)
            else:
                # logging.debug(f" For camera {self.camera_id} has frames, will process now")
                while self.video_stream.more():
                    try:
                        original_frame = self.video_stream.read()
                        self.motion_file_processor.take(original_frame=original_frame)

                        if self.output_motion_video:
                            if self.video_start_time is not None:
                                diff_time = datetime.now() - self.video_start_time
                                if diff_time >= timedelta(minutes=5):
                                    self.video_writer.release()
                                    self.video_writer = None
                                    self.video_start_time = None

                            if self.video_writer is None:
                                self.video_start_time = datetime.now()
                                frame_width = int(self.video_stream.get_width())
                                frame_height = int(self.video_stream.get_height())
                                frame_size = (frame_width, frame_height)
                                fps = int(self.video_stream.get_fps())
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
        print(f"thread stopped for cameraid {self.camera_id}")
        self.motion_file_processor.stop()
        self.video_stream.stop()
        self.video_writer.release()
        self.stopped = True
