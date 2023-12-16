import logging
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2

import MOG2
import RegionOfInterest
import VideoStreamWriter
from RtspVideoStream import RtspVideoStream


class CameraMotion:
    def __init__(self, camera_conf_name, camera_id, parser):
        self.video_writer = None
        self.t = None
        self.stopped = False
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
        self.video_stream = RtspVideoStream(self.video_url).start()
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
        self.t.daemon = True
        self.t.start()
        return self

    def join(self):
        self.t.join()
        return self

    @staticmethod
    def motion_not_detected(event_path, camera_id):
        end_time_formatted = camera_id + "_MotionEnd_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".txt"
        logging.debug("event ended {date_time}".format(date_time=end_time_formatted))
        open(event_path + end_time_formatted, 'w')

    @staticmethod
    def motion_detected(event_path, camera_id):
        start_time_formatted = camera_id + "_MotionStart_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".txt"
        logging.debug("event started {date_time}".format(date_time=start_time_formatted))
        open(event_path + start_time_formatted, 'w')

    def process(self):
        while not self.stopped:
            if not self.video_stream.more():
                logging.debug(f" For camera {self.camera_id} no more frames, waiting for 2 seconds")
                time.sleep(2.0)
            else:
                logging.debug(f" For camera {self.camera_id} has frames, will process now")
                while self.video_stream.more():
                    original_frame = self.video_stream.read()
                    if self.output_motion_video:
                        if self.video_writer is None:
                            frame_width = int(self.video_stream.get_width())
                            frame_height = int(self.video_stream.get_height())
                            frame_size = (frame_width, frame_height)
                            fps = int(self.video_stream.get_fps())
                            video_file_output = self.event_path + 'output.mp4'
                            self.video_writer = cv2.VideoWriter(video_file_output,
                                                                cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                                                                frame_size)
                    try:
                        # frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
                        # frame = cv2.GaussianBlur(frame, (int(self.blur), int(self.blur)), 0)
                        # frame = RegionOfInterest.mask(frame, self.regions)
                        frame = original_frame
                        final_frame = self.mog2.apply(frame)
                        # Finding contour of moving object
                        if final_frame is not None:
                            contours, _ = cv2.findContours(final_frame.copy(),
                                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            contours = sorted(contours, key=cv2.contourArea, reverse=True)

                            has_motion = []
                            contours_filtered = []
                            for contour in contours:
                                if cv2.contourArea(contour) < self.area:
                                    has_motion.append(False)
                                else:
                                    contours_filtered.append(contour)
                                    has_motion.append(True)

                            contour_has_motion = any(has_motion)
                            # if contour_has_motion:
                                # cv2.putText(original_frame,
                                #             'Motion Detected' + datetime.now().strftime("%m-%d-%Y_%H:%M:%S"),
                                #             (20, 300), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255, 2))
                                # cv2.drawContours(image=original_frame, contours=contours_filtered, contourIdx=-1,
                                #                  color=255,
                                #                  thickness=3)

                            if len(has_motion) > 0:
                                if contour_has_motion and self.detect_time is None:
                                    self.detect_time = datetime.now()
                                    self.end_time = None
                                    self.motion_detected(self.event_path, self.camera_id)
                                if contour_has_motion and self.detect_time is not None:
                                    self.detect_time = datetime.now()
                                    self.end_time = None
                                else:
                                    # we do not have any motion
                                    if self.end_time is None and self.detect_time is not None:
                                        self.end_time = datetime.now()
                            else:
                                # we do not have any motion
                                if self.end_time is None and self.detect_time is not None:
                                    self.end_time = datetime.now()

                            if self.end_time is not None:
                                new_end_time = self.end_time + timedelta(seconds=int(self.post_motion_wait))
                                diff = new_end_time - datetime.now()
                                # cv2.putText(original_frame,
                                #             'Time Elapsed Post Motion End {time}'.format(time=diff / 1000),
                                #             (20, 250),
                                #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255, 2))
                                if new_end_time < datetime.now():
                                    self.motion_not_detected(self.event_path, self.camera_id)
                                    self.end_time = None
                                    self.detect_time = None
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
        self.video_stream.stop()
        self.video_writer.release()
        self.stopped = True
