import time
from PIL import Image
import numpy as np
# Local libraries
import utils

# Constants
from constants import MAX_COLOR_VALUE
NORMAL_MAP_FILENAME = "normal.jpg"
OUTPUT_FILENAME = "img_out.png"
L = utils.normalize(np.array([1, 1, 1]))

def shade(n, l):
    """
    Shader calculation for a normal and a light vector.
    Args:
        n(numpy.array): Unit normal vector
        l(numpy.array): Unit vector in the direction to the light
    Returns:
        numpy.uint8: The calculated color (grayscale 0-255)
    """
    diffuse_coef = (np.dot(n, l) + 1) / 2
    color = np.uint8(np.maximum(0, diffuse_coef) * MAX_COLOR_VALUE)
    return color

def main():
    start = time.time()
    print("Opening Normal Map...")
    im = Image.open(NORMAL_MAP_FILENAME)
    # Create an array from the image
    im_array = np.asarray(im)
    # Create output image vector
    w, h, channels = im_array.shape
    output = np.zeros((w, h), dtype=np.uint8)
    print("Shading image...")
    # Iterate over the array
    for i in range(w):
        for j in range(h):
            normal_map_pixel = im_array[j][i]
            n = utils.adjust(normal_map_pixel)
            output[j][i] = shade(n, L)
    # Turn output into image and show it
    im_output = Image.fromarray(output)
    im_output.save(OUTPUT_FILENAME)
    print("Output image saved as {}".format(OUTPUT_FILENAME))
    end = time.time()
    elapsed_time = (end - start)
    print("Elapsed time was: {}ms".format(elapsed_time * 1000))
    print("or in human time: {}".format(utils.humanize_time(elapsed_time)))



if __name__ == '__main__':
    main()
