# -*- coding: utf-8 -*-
import cv2 
import requests
import numpy as np
import re


def load_avi(file_name):
    """Reads in .avi file and returns list of 3 channel images."""
    a = []
    b = []
    frames = []
    with open(file_name) as f:
        bytes = f.read()
    p = re.compile(r'\xff\xd8')
    p2 = re.compile(r'\xff\xd9')
    for m in p.finditer(bytes):
        a += [m.start()]
    for m in p2.finditer(bytes):
        b += [m.start()]
    for c, d in zip(a, b):
        jpg = bytes[c:d+2]
        frames.append(cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),
                                   cv2.CV_LOAD_IMAGE_COLOR))
    return frames


def get_feed(url=None):
    """ Get video feed from webcam unless url of IP camera is provided."""
    if url is None:
        feed = cv2.VideoCapture(0)
    else:
        feed = requests.get(url, stream=True)
    return feed


def get_frame_ip(feed):
    """"Get frame from video feed of IP camera."""
    bytes = ''
    bytes += feed.raw.read(80000)
    a = bytes.find('\xff\xd8')
    b = bytes[a:].find('\xff\xd9')+a

    if a != -1 and b != -1:
        jpg = bytes[a:b+2]
        frame = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),
                             cv2.CV_LOAD_IMAGE_COLOR)
    else:
        frame = None
    return frame


def get_frame_cam(feed):
    """Get frame of video from video feed of webcam""" 
    if_read, frame = feed.read()
    assert if_read, 'Frame not read'
    return frame


def get_frame(feed): 
    """Get frame of video from webcam or IP camera feed"""
    if isinstance(feed, str):
        frame = get_frame_ip(feed)
    else:
        frame = get_frame_cam(feed)
    return frame


def to_gray(frame):
    """ Convert frame to one channel using cv2.COLOR_BGR2GRAY."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray_frame


def to_red(frame):
    """Extract red channel from frame."""
    b, g, r = cv2.split(frame)
    return r


def to_green(frame):
    """Extract green channel from frame."""
    b, g, r = cv2.split(frame)
    return g


def to_blue(frame):
    """Extract blue channel from frame."""
    b, g, r = cv2.split(frame)
    return b    


def to_pca(frame):
    """ Converts frame to one channel using principal component analysis."""
    m, n, p = np.shape(frame)
    x = np.reshape(frame, (m * n, 3))
    x_std = np.true_divide(x - x.mean(), 256)
    cov_mat = np.cov(x_std.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i])
                 for i in range(len(eig_vals))]
    eig_pairs.sort()
    eig_pairs.reverse()
    matrix_w = eig_pairs[0][1].reshape(3, 1)
    y = x_std.dot(matrix_w)
    matrix = np.true_divide(255, y.max() - y.min())
    b = -matrix * y.min()
    y_new = 255 - (y * matrix + b)
    y_new = y_new.astype('uint8')
    pca = y_new.reshape((m, n))
    return pca


def get_running_avg(colored_frame, running_avg=None, g=41, alpha=0.02,
                    color_convert='gray'):
    """Convert frame to one channel, smooth, and update running average.
    
    Args:
        colored_frame -- 3 channel image
        running_avg (optional) -- 1 channel average (default is None)
        g (optional odd int) -- # of pixels to consider in the x and y 
            direction for Gaussian filter (default is 31)
        alpha (optional float) -- regulates how fast the accumulator 
             forgets about earlier images for calculating the running_avg
        color_convert (optional string) -- option for how to convert to one
            channel. Options are gray, blue, green, red, pca (default 'gray')
    
    Returns: 
        smooth -- 1 channel smoothed image
        running_avg -- 1 channel running average of previous images
    """ 
    color_converter = dict(gray=to_gray, red=to_red, green=to_green,
                           blue=to_blue, pca=to_pca)
                           
    one_channel = color_converter[color_convert](colored_frame)
    
    smooth = cv2.GaussianBlur(one_channel, (g, g), 0)
    
    if running_avg is None:
        running_avg = np.float32(smooth)
    else:
        cv2.accumulateWeighted(np.float32(smooth), running_avg,
                               alpha, None)
    return smooth, running_avg


def get_diff(frame, running_avg, thresh=35):
    """Gets difference between current frame and weighted average as well as 
    a binary image created from a threshold of the difference.
    
    Args:
        frame -- one channel image
        running_avg -- one channel running average image 
        thresh (optional) -- any pixels in differce greater than thresh will
            be considered movement (default 35).
    
    Returns: 
        diff -- absolute difference between the frame and running_avg
        thresh_diff -- 0 where diff>thresh, 255 where diff<thresh
    """
    
    diff = cv2.absdiff(np.float32(frame), np.float32(running_avg))
    _, thresh_diff = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY_INV)
    return diff, thresh_diff


def left_minus_right(frame, midpt=320, upper_thresh=1000000,
                     lower_thresh=-400000):
    """Subtracts the sum of the frame's right side pixel values from the sum
    of the left side pixel values and then categorizes the frame according
    to if it is less than lower_thresh, between mid_tresh and upper_thresh, or
    above upper_thresh (signifying more motion on left than right, similar
    motion, and more motion on right than left).

    :param frame: one channel image
    :param midpt: int - the x coordinate dividing the two sides to be sum.
        Default = 320
    :param upper_thresh: float or int - if th
    :param lower_thresh:
    :param mid_thresh:
    :return:
    """
    left = sum(sum(frame[:, :midpt]))
    right = sum(sum(frame[:, midpt:]))
    left_minus_right = left-right

    if left_minus_right < lower_thresh:
        there = 'left'
    elif left_minus_right > upper_thresh:
        there = 'train'
    else:
        there = 'none'
        
    return there, left_minus_right


def detect_train(train_count, there, time, min_count=50):
    """Looks at cumulative list of train classifications (train_count)
    and current classification to determine if a train just passed.

    If the current frame has no train detected (i.e. there = 'none) but
    the last min_count frames have resulted in a train being detected
    (either fully - there = 'train', or partially - there = 'left' or 'right',
    we say that a train has passed.

    If 'left' or 'right' is at the beginning or end of that list, we can
    predict the direction in which it moved (left half at the beginning means
    it is south bound).

    We return (time, direction) as train_pass and empty the train_count list
    to start the process over again for the next train.

    If the current frame has no train detected but there is a list already
    begun, we give the classifier the benefit of the chance and append to the
    current list.

    If current frame has not train detected and the end of train_count is
    'none' (i.e we gave it the benefit of the chance before), then we
    empty the list and confirm there was no train - must have been other
    movement.

    If current frame has left, right or train detected, we append it to
    train_count.

    This seems annoyingly complicated and a simpler approach is welcome!

    :param train_count: A list of previously detected movement. Will be
        empty if there has been nothing detected for >1 frame.
    :param there: string - 'left' if left side has more movement than right
        side, 'right' for the opposite, 'train' if movement across image,
        'none' otherwise.
    :param time: anything, some indication of what frame is current.
    :param min_count: how many screens to count as the train - for mpeg, more
        than 50 is likely.
    :return train_pass: (time, direction) if train. Otherwise None
    :return train_count: if train detected, there is appended
    """
    train = None

    if there != 'none':
        train_count.append(there)
    elif len(train_count) == 0:
        train_count = []
    elif len(train_count) < min_count:
        if train_count[-1] == 'none':
            train_count = []
    elif len(train_count) > min_count:
        if train_count[0] == 'left':
            train = (time, 'south')
            train_count = []
        else:
            train = (time, 'north')
            train_count = []
    else:
        train_count.append(there)
    return train, train_count

def train_detector(frames, running_avg=None, train_count=None, 
                 train_pass=None, play=True, alpha=0.2, 
                 g=31, thresh=25, lower_thresh=-400000, 
                 upper_thresh=1000000, times=None, min_count=45):
    
    if train_count is None:
        train_count = []
    if train_pass is None:
        train_pass = []
    if times is None:
        times = range(len(frames))
    for j,frame in enumerate(frames):
        smooth_frame, running_avg = get_running_avg(frame, running_avg,
                                                       alpha=alpha, g=g)
        diff, thresh_diff = get_diff(smooth_frame, running_avg,
                                        thresh=thresh)
        there, l_m_r= left_minus_right(thresh_diff, upper_thresh=upper_thresh,lower_thresh=lower_thresh)
        train, train_count = detect_train(train_count, there, times[j], min_count=min_count)
        
        if play:
            cv2.imshow('Original video', frame)

        # Add newly detected train to list
        if train is not None:
            train_pass.append(train)
            train = None

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    if play:        
        cv2.destroyWindow('Original video')
    return train_pass, train_count, running_avg
# Following functions are not used in train detection.

def get_edges_contours(frame, lower=10, upper=20):
    """ Returns edges and contours. Contours returned are only those highest
    in the hierarchy. Inner contours are not returned.

    Args:
        frame -- one channel image
        lower (optional int) -- lower threshold for the pixel gradient value
            calculated in the Canny algorithm. Pixels below this value will
            be rejected (default 10).
        upper (optional int) -- upper threshold for pixel gradient value.
            Pixels above this value will be considered edges (default 20).

    Returns:
        edges -- edges detected using Canny algorithm.
        contours -- the most external contours found from the edges.
    """
    edges = cv2.Canny(np.uint8(frame), lower, upper)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    return edges, contours


def get_centroids(contours, min_area=1000, frame=None, circle_radius=10):
    """ Get the centroid of the contours for contours that meet a threshold.

    Args:
        contours -- contours found from image
        min_area (optional int) -- minimum area of closed contours to be
            considered in finding centroids (default 1000).
        frame (optional) -- image. If provided, will add circles at the
            locations of the centroids (default None).
        circle_radius (optional int) - number of pixels for radius of circle
            drawn on image if provided at centroid locations (default 10).

    Return:
        centroids -- list of tuples (x,y) for centroid locations
        areas -- list of floats for area of closed contour
        frame -- image with drawn centroids (if initially provided)
    """
    x_centroids = []
    y_centroids = []
    areas = []
    frame = np.uint8(frame)

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > min_area:
            M = cv2.moments(contour)
            if M['m00'] !=0:
                cx = (int(M['m10']/M['m00']))
                cy = (int(M['m01']/M['m00']))
                if frame is not None:
                    cv2.circle(frame, (cx, cy), circle_radius,
                               (76, 20, 255), -1)
                x_centroids.append(cx)
                y_centroids.append(cy)
            areas.append(area)

    return x_centroids, y_centroids, areas, frame