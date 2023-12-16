import cv2
import logging


def create(parser, event_path, fps, width, height):
    output_motion_video = parser.getboolean("DEFAULT", "output_motion_video")
    if output_motion_video:
        fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')
        video_file_output = event_path + 'output.mp4'
        logging.debug(f"video_writer created {video_file_output}")
        video_writer = cv2.VideoWriter(video_file_output, fourcc, fps, (width, height))
        return video_writer
    else:
        return None
