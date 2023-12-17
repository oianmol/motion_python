import logging
import threading
import time
from datetime import datetime, timedelta
from queue import Queue

import cv2
import numpy as np

import RegionOfInterest

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )


class MotionFileProcessor:
    def __init__(self):
        self.stopped = False

    def take(self, video_file_path, blur, regions, area, camera_id, event_path, post_motion_wait, mog2):
        t = threading.Thread(target=self.process_video_new, args=(
            video_file_path, blur, regions, area, camera_id, event_path, post_motion_wait, mog2))
        t.start()
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
        self.stopped = True

    def process_video(self, video_file_path, blur, regions, area, camera_id, event_path, post_motion_wait, mog2):
        end_time = None
        detect_time = None
        start_time = datetime.now()
        track_len = 10
        detect_interval = 5
        tracks = []
        frame_idx = 0
        file_stream = cv2.VideoCapture(video_file_path)

        while not self.stopped:
            (grabbed, original_frame) = file_stream.read()
            time.sleep(0.0008)
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
                if end_time is not None or detect_time is not None:
                    self.motion_not_detected(event_path, camera_id)
                logging.debug(f"Finished processing {video_file_path} {datetime.now() - start_time}")
                break
        if file_stream is not None:
            file_stream.release()
            print(f"file stream released")

    def process_video_new(self, video_file_path, blur, regions, area, camera_id, event_path, post_motion_wait, mog2):
        end_time = None
        detect_time = None
        prev_gray = None
        start_time = datetime.now()
        track_len = 10
        detect_interval = 5
        tracks = []
        frame_idx = 0
        file_stream = cv2.VideoCapture(video_file_path)

        while not self.stopped:
            (grabbed, original_frame) = file_stream.read()
            time.sleep(0.0008)
            if grabbed and original_frame is not None:
                frame_gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
                vis = original_frame.copy()

                if len(tracks) > 0:
                    img0, img1 = prev_gray, frame_gray
                    p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
                    p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                    p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                    d = abs(p0 - p0r).reshape(-1, 2).max(-1)
                    good = d < 1
                    new_tracks = []
                    for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
                        if not good_flag:
                            continue
                        tr.append((x, y))
                        if len(tr) > track_len:
                            del tr[0]
                        new_tracks.append(tr)
                    tracks = new_tracks
                    cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))

                if frame_idx % detect_interval == 0:
                    mask = np.zeros_like(frame_gray)
                    mask[:] = 255
                    p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
                    if p is not None:
                        for x, y in np.float32(p).reshape(-1, 2):
                            tracks.append([(x, y)])

                if len(tracks) > 0:
                    if detect_time is None:
                        detect_time = datetime.now()
                        end_time = None
                        self.motion_detected(event_path, camera_id)
                    if detect_time is not None:
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
                    if new_end_time < datetime.now():
                        self.motion_not_detected(event_path, camera_id)
                        end_time = None
                        detect_time = None

                frame_idx += 1
                prev_gray = frame_gray
            else:
                if end_time is not None or detect_time is not None:
                    self.motion_not_detected(event_path, camera_id)
                logging.debug(f"Finished processing {video_file_path} {datetime.now() - start_time}")
                break
        if file_stream is not None:
            file_stream.release()
            print(f"file stream released")
