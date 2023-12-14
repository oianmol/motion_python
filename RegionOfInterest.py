from collections import namedtuple
import typing as ty
import logging
import numpy as np
import cv2

Point = namedtuple("Point", ['x', 'y'])
Size = namedtuple("Size", ['w', 'h'])


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


def prepare(region_of_interest, width, height):
    regions = []
    if len(region_of_interest) > 0:
        x = region_of_interest.split(" ")
        it = iter(list(map(int, x)))
        for x in it:
            regions.append(Point(x, next(it)))
    if len(regions) == 0:
        initial_region = [initial_point_list(w=width, h=height)]
        regions = initial_region
    return regions


def mask(frame, regions):
    mask = np.zeros_like(frame, dtype=np.uint8)
    for shape in [regions]:
        points = np.array([shape], np.int32)
        mask = cv2.fillPoly(mask, points, color=(255, 255, 255), lineType=cv2.LINE_4)
    frame = np.bitwise_and(frame, mask).astype(np.uint8)
    return frame
