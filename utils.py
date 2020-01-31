import numpy as np

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def adjust(normal_map_pixel):
    """
    Adjust the normal map pixel so that it can have negative values.

    Args:
        normal_map_pixel(numpy.array): Three channels (B,G,R) pixel
            representing a normal vector.

    Returns:
        numpy.array: A ready to use normal vector
    """
    # Change B,G,R to R,G,B
    x, y, z = normal_map_pixel
    x -= 128
    y -= 128
    normal_map_pixel = np.array((x, y, z)) / 255.0
    return normalize(normal_map_pixel)
