import numpy as np
from progress.bar import Bar
import time


from constants import MAX_COLOR_VALUE


def normalize(arr):
    """
    Normalize a vector using numpy.
    Args:
        arr(darray): Input vector
    Returns:
        darray: Normalized input vector
    """
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr
    return arr / norm


def distance(p1, p2):
    """
    Get the distance between points p1 and p2
    Args:
        p1(ndarray): Point 1
        p2(ndarray): Point 2
    Returns:
         float: Distance
    """
    dist = np.linalg.norm(p1 - p2)
    return dist


def humanize_time(secs):
    mins, secs = divmod(secs, 60)
    hours, mins = divmod(mins, 60)
    return '%02d:%02d:%02d' % (hours, mins, secs)


def degrees2radians(degrees):
    return (degrees / 360) * 2 * np.pi


def normalize_color(color):
    return color / MAX_COLOR_VALUE


def blerp(img_arr, x, y):
    # Interpolate values of pixel neighborhood of x and y
    i = int(np.round(x))
    j = int(np.round(y))
    # But not in the borders
    height, width, _ = img_arr.shape
    # Flip y value to go from top to bottom
    y = height - y
    if i == 0 or j == 0 or i == width or j == height:
        if i == width:
            i -= 1
        if j == height:
            j -= 1
        return img_arr[j][i]
    # t and s are interpolation parameters that go from 0 to 1
    t = x - i + 0.5
    s = y - j + 0.5
    # Bilinear interpolation
    color = (
        img_arr[j - 1][i - 1] * (1 - t) * (1 - s)
        + img_arr[j - 1][i] * t * (1 - s)
        + img_arr[j][i - 1] * (1 - t) * s
        + img_arr[j][i] * t * s
    )
    return color


def adjust(normal_map_pixel):
    """
    Adjust the normal map pixel so that it can have negative values.

    Args:
        normal_map_pixel(ndarray): Three channels RGB pixel
            representing a normal vector.

    Returns:
        ndarray: A ready to use unit normal vector
    """
    r, g, b = normal_map_pixel
    # x and y will be from [-1 to 1] and z from [0 to 1]
    x = 2 * r - 1
    y = 2 * g - 1
    z = b
    normal_vector = np.array([x, y, z])
    normalized = normalize(normal_vector)
    return normalized


def progress_bar(iterations, msg):
    step_size = np.round(iterations / 100).astype(int)
    bar = Bar(msg, max=100, suffix='%(percent)d%%')
    return bar, step_size


class Timer:
    def __init__(self):
        self.start_time = 0
        self.end_time = 0
        self.elapsed_time = 0

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time

    def __str__(self):
        return humanize_time(self.elapsed_time)
