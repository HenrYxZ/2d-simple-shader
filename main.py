from PIL import Image
import numpy as np
# Local libraries
import utils

# Constants
NORMAL_MAP_FILENAME = "normal.jpg"
L = utils.normalize(np.array([1, 1, 1]))
MAX_COLOR_VALUE = 255


def main():
    im = Image.open(NORMAL_MAP_FILENAME)
    # Create an array from the image
    im_array = np.asarray(im)
    # Create output image vector
    w, h, channels = im_array.shape
    output = np.zeros((w, h))
    # Iterate over the array
    counter = 0
    for i in range(w):
        for j in range(h):
            normal_map_pixel = im_array[j][i]
            n = utils.adjust(normal_map_pixel)
            n_dot_L = np.dot(n, L)
            output[j][i] = np.maximum(0, n_dot_L) * MAX_COLOR_VALUE
            if counter % 1000 == 0:
                print(
                    "Iteration number {0}, i={1}, j={2}".format(counter, i, j)
                )
            counter += 1
    # Turn output into image and show it
    im_output = Image.fromarray(output)
    im_output.show()


if __name__ == '__main__':
    main()
