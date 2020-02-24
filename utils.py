import numpy as np
# Local Libraries
from constants import MAX_COLOR_VALUE


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def humanize_time(secs):
    """
    Extracted from http://testingreflections.com/node/6534
    """
    mins, secs = divmod(secs, 60)
    hours, mins = divmod(mins, 60)
    return '%02d:%02d:%02f' % (hours, mins, secs)


def adjust(normal_map_pixel):
    """
    Adjust the normal map pixel so that it can have negative values.

    Args:
        normal_map_pixel(numpy.array): Three channels (B,G,R) pixel
            representing a normal vector.

    Returns:
        numpy.array: A ready to use normal vector
    """
    r, g, b = normal_map_pixel
    # x and y will be from [-1 to 1] and z from [0 to 1]
    x = 2 * (r / np.float(MAX_COLOR_VALUE)) - 1
    y = 2 * (g / np.float(MAX_COLOR_VALUE)) - 1
    z = b / float(MAX_COLOR_VALUE)
    normal_vector = np.array([x, y, z])
    return normalize(normal_vector)
