import logging
import threading
import time
from datetime import datetime, timedelta
from queue import Queue

import cv2

import RegionOfInterest


class MotionFileProcessor:
    def __init__(self):
        self.stopped = False
        self.processing_queue = Queue(maxsize=150)  # or total cameras

    def start(self):
        # start a thread to read frames from the file video stream
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def take(self, video_file_path, blur, regions, area, camera_id, event_path, post_motion_wait, mog2):
        if not self.processing_queue.full():
            logging.debug(f"added video to queue {video_file_path} for cam id {camera_id}")
            self.processing_queue.put(
                [video_file_path, blur, regions, area, camera_id, event_path, post_motion_wait, mog2])
        else:
            logging.error(f"Queue full cannot add more videos for processing. {self.processing_queue.qsize()}")
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

    def read(self):
        # return next frame in the queue
        return self.processing_queue.get()

    def more(self):
        # return True if there are still frames in the queue
        return self.processing_queue.qsize() > 0

    def stop(self):
        self.stopped = True

    def update(self):
        while not self.stopped:
            while self.more():
                [video_file_path, blur, regions, area, camera_id, event_path, post_motion_wait, mog2] = self.read()
                logging.debug(f"Pending videos in queue {self.processing_queue.qsize()}")
                end_time = None
                detect_time = None
                start_time = datetime.now()
                file_stream = cv2.VideoCapture(video_file_path)
                while True:
                    (grabbed, original_frame) = file_stream.read()
                    if grabbed and original_frame is not None:
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
                                cv2.drawContours(image=original_frame, contours=contours_filtered, contourIdx=-1,
                                                 color=255,
                                                 thickness=3)

                            if len(has_motion) > 0:
                                if contour_has_motion and detect_time is None:
                                    detect_time = datetime.now()
                                    end_time = None
                                    self.motion_detected(event_path, camera_id)
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
                                cv2.putText(original_frame,
                                            'Time Elapsed Post Motion End {time}'.format(time=diff / 1000),
                                            (20, 250),
                                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255, 2))
                                if new_end_time < datetime.now():
                                    self.motion_not_detected(event_path, camera_id)
                                    end_time = None
                                    detect_time = None
                    else:

                        logging.debug(f"Finished processing {video_file_path} {datetime.now() - start_time}")
                        break
