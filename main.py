import time
from PIL import Image
import numpy as np
import os.path
# Local libraries
import utils

# Constants
from constants import MAX_COLOR_VALUE
NORMAL_MAP_FILENAME = "normal.jpg"
OUTPUT_FILENAME = "img_out.png"
NORMAL_VECTORS_FILENAME = "normals.npy"
SECOND_TO_MS = 1000
NORMAL_DIMENSIONS = 3
# The program will output an update every (this number) percent done
PERCENTAGE_STEP = 10
L = utils.normalize(np.array([1, 1, 1]))


def adjust_normal_map(rgb_normal_map):
    """
    Return an adjusted normal map on which each element is a normalized vector
    from a RGB normal map.
    Args:
        rgb_normal_map(numpy.array): The RGB normal map as a numpy array.
    Returns:
        numpy.array: Map of normalized vector normals.
    """
    print("Creating normal vectors from RGB map...")
    w, h, _ = rgb_normal_map.shape
    iterations = w * h
    step_size = np.ceil((iterations * PERCENTAGE_STEP) / 100).astype('int')
    normals = np.zeros((w, h, NORMAL_DIMENSIONS))
    counter = 0
    for i in range(w):
        for j in range(h):
            if counter % step_size == 0:
                percent_done = int((counter / float(iterations)) * 100)
                print("{}% of normal vectors created".format(percent_done))
            normals[j][i] = utils.adjust(rgb_normal_map[j][i])
            counter += 1
    return normals


def shade(n, l):
    """
    Shader calculation for a normal and a light vector.
    Args:
        n(numpy.array): Unit normal vector
        l(numpy.array): Unit vector in the direction to the light
    Returns:
        numpy.uint8: The calculated color (grayscale 0-255)
    """
    # This formula changes the value [-1 - 1] to [0 - 1]
    diffuse_coef = (np.dot(n, l) + 1) / 2
    color = np.uint8(np.maximum(0, diffuse_coef) * MAX_COLOR_VALUE)
    return color


def use_normal_map():
    start = time.time()
    print("Opening Normal Map...")
    normal_im = Image.open(NORMAL_MAP_FILENAME)
    # Create an array from the image
    normal_im_array = np.asarray(normal_im)
    if os.path.exists("normals.npy"):
        # Load normals from file
        print("Loading normal vectors from file {}".format(NORMAL_VECTORS_FILENAME))
        normals = np.load(NORMAL_VECTORS_FILENAME)
    else:
        # Create the normals vector map
        start_normals = time.time()
        normals = adjust_normal_map(normal_im_array)
        np.save(NORMAL_VECTORS_FILENAME, normals)
        print(
            "Normal vectors stored inside {} file".format(
                NORMAL_VECTORS_FILENAME
            )
        )
        end_normals = time.time()
        elapsed_time = utils.humanize_time(end_normals - start_normals)
        print("Time adjusting normals was: {}".format(elapsed_time))
    # Create output image vector
    w, h, _ = normal_im_array.shape
    output = np.zeros((h, w), dtype=np.uint8)
    print("Shading image...")
    start_shading = time.time()
    # Iterate over the array
    for i in range(w):
        for j in range(h):
            n = normals[j][i]
            output[j][i] = shade(n, L)
    end_shading = time.time()
    elapsed_time = utils.humanize_time(end_shading - start_shading)
    print("Time shading was: {}".format(elapsed_time))
    # Turn output into image and show it
    im_output = Image.fromarray(output)
    im_output.save(OUTPUT_FILENAME)
    print("Output image saved as {}".format(OUTPUT_FILENAME))
    end = time.time()
    elapsed_time = (end - start)
    print("Elapsed time was: {}ms".format(elapsed_time * SECOND_TO_MS))
    print("or in human time: {}".format(utils.humanize_time(elapsed_time)))


def main():
    use_normal_map()


if __name__ == '__main__':
    main()
