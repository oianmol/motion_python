import logging
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
import cv2
import RegionOfInterest


class MotionFileProcessor:
    def __init__(self, blur, regions, area, camera_id, event_path, post_motion_wait, mog2):
        self.blur = blur,
        self.regions = regions
        self.area = area
        self.camera_id = camera_id
        self.event_path = event_path
        self.post_motion_wait = post_motion_wait
        self.all_files = []
        self.mog2 = mog2

    def take(self, video_file_path):
        t = threading.Thread(target=self.update, args=(video_file_path,))
        t.daemon = True
        t.start()
        self.all_files.append(t)
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

    def stop(self):
        logging.debug(f"stopped all motion processing threads for {self.camera_id}")
        for file_thread in self.all_files:
            file_thread.stop()

    def update(self, video_file_path):
        end_time = None
        detect_time = None
        file_stream = cv2.VideoCapture(video_file_path)
        while True:
            (grabbed, original_frame) = file_stream.read()
            time.sleep(0.250) ## delay for FPS since it's 4.0
            if grabbed and original_frame is not None:
                frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.GaussianBlur(frame, (int(self.blur[0]), int(self.blur[0])), 0)
                frame = RegionOfInterest.mask(frame, self.regions)
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
                    if contour_has_motion:
                        cv2.putText(original_frame,
                                    'Motion Detected' + datetime.now().strftime("%m-%d-%Y_%H:%M:%S"),
                                    (20, 300), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255, 2))
                        cv2.drawContours(image=original_frame, contours=contours_filtered, contourIdx=-1,
                                         color=255,
                                         thickness=3)

                    if len(has_motion) > 0:
                        if contour_has_motion and detect_time is None:
                            detect_time = datetime.now()
                            end_time = None
                            self.motion_detected(self.event_path, self.camera_id)
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
                        new_end_time = end_time + timedelta(seconds=int(self.post_motion_wait))
                        diff = new_end_time - datetime.now()
                        cv2.putText(original_frame,
                                    'Time Elapsed Post Motion End {time}'.format(time=diff / 1000),
                                    (20, 250),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255, 2))
                        if new_end_time < datetime.now():
                            self.motion_not_detected(self.event_path, self.camera_id)
                            end_time = None
                            detect_time = None
            else:
                logging.debug(f"Finished processing {video_file_path}")
                break
