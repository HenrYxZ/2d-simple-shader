import time
from PIL import Image
import numpy as np
import os.path
# Local libraries
import utils

# Constants
from constants import MAX_COLOR_VALUE

NORMAL_MAP_FILENAME = "normal.jpg"
OUTPUT_FILENAME = "img_out.jpg"
NORMAL_VECTORS_FILENAME = "normals"
NORMAL_VECTORS_FILE_EXT = ".npy"
SECOND_TO_MS = 1000
NORMAL_DIMENSIONS = 3
DARK_IMG_FILENAME = "dark.jpg"
LIGHT_IMG_FILENAME = "light.jpg"
SPEC_IMG_FILENAME = "spec.jpg"
RGB_CHANNELS = 3
DEFAULT_NORMALS_SIZE = 512
CREATED_NORMALS_FILENAME = "created_normals.npy"
CREATED_NORMALS_IMG_FILENAME = "created_normals.jpg"
LIGHT_COLOR = np.array([232, 158, 39])
DARK_COLOR = np.array([14, 5, 74])
COLOR_FOR_LIGHT = np.array([255, 255, 255])
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
    normals = np.zeros((h, w, NORMAL_DIMENSIONS))
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
    # diffuse_coef = (np.dot(n, l) + 1) / 2
    diffuse_coef = np.dot(n, l)
    color = np.uint8(np.maximum(0, diffuse_coef) * MAX_COLOR_VALUE)
    return color


def shade_colors(n, l, dark, light):
    """
    Shader calculation for a normal and a light vector and light and dark
    colors.
    Args:
        n(numpy.array): Unit normal vector
        l(numpy.array): Unit vector in the direction to the light
        dark(numpy.array): RGB dark color
        light(numpy.array): RGB light color
    Returns:
        numpy.uint8: The calculated color (RGB)
    """
    # This formula changes the value [-1 - 1] to [0 - 1]
    diffuse_coef = np.dot(n, l)
    t = np.maximum(0, diffuse_coef)
    color = light * (1 - t) + dark * t
    return color


def shade_with_specular(n, l, dark, light, ks):
    """
    Shader calculation for normal and light vectors and dark, light and
    specular colors.
    Args:
        n(numpy.array): Unit normal vector
        l(numpy.array): Unit vector in the direction to the light
        dark(numpy.array): RGB dark color
        light(numpy.array): RGB light color
        ks(float): size of specularity (this can be changed by the user)

    Returns:
        numpy.uint8: The calculated color (RGB)
    """
    n_dot_l = np.dot(n, l)
    t = np.maximum(0, n_dot_l)
    color = light * t + dark * (1 - t)
    # --------------- Adding specular
    # unit vector in the direction to the eye or viewer
    eye = np.array([0, 0, 1])
    # reflection of the light in the given point
    # r = l * - 1 + 2 * n_dot_l * n
    # s = np.dot(eye, r)
    s = l[2] * -1 + 2 * n[2] * n_dot_l
    s = np.maximum(0, s)
    # alpha = 1
    # s = s ** alpha
    color = color * (1 - s * ks) + s * ks * COLOR_FOR_LIGHT
    return color


def create_normal_map():
    print("Creating normal map...")
    # iterate 512x512 array
    normals = np.zeros((
        DEFAULT_NORMALS_SIZE, DEFAULT_NORMALS_SIZE, NORMAL_DIMENSIONS
    ))
    x0 = DEFAULT_NORMALS_SIZE / 2 - 1
    y0 = DEFAULT_NORMALS_SIZE / 2 - 1
    iterations = DEFAULT_NORMALS_SIZE ** 2
    step_size = np.ceil((iterations * PERCENTAGE_STEP) / 100).astype('int')
    counter = 0
    for i in range(DEFAULT_NORMALS_SIZE):
        for j in range(DEFAULT_NORMALS_SIZE):
            # create [j][i] normal
            x = i - x0
            y = j - y0
            z = np.sqrt(DEFAULT_NORMALS_SIZE ** 2 - np.absolute(x * y))
            normals[j][i] = np.array([x, y, z]) * MAX_COLOR_VALUE
            counter += 1
            if counter % step_size == 0:
                percent_done = int((counter / float(iterations)) * 100)
                print("{}% of normal map created".format(percent_done))
    normals_array = normals.astype(np.uint8)
    normals_img = Image.fromarray(normals_array)
    normals_img.save("created_normals.png")
    return normals_img


def use_normal_map(normal_img, normal_opt):
    # Create an array from the image
    normal_im_array = np.asarray(normal_img)
    normals_filename = (
            NORMAL_VECTORS_FILENAME + normal_opt + NORMAL_VECTORS_FILE_EXT
    )
    if os.path.exists(normals_filename):
        # Load normals from file
        print(
            "Loading normal vectors from file {}".format(
                normals_filename
            )
        )
        normals = np.load(normals_filename)
        w, h, _ = normals.shape
        return normals, w, h
    # Create the normals vector map
    start_normals = time.time()
    normals = adjust_normal_map(normal_im_array)
    np.save(normals_filename, normals)
    print(
        "Normal vectors stored inside {} file".format(
            normals_filename
        )
    )
    end_normals = time.time()
    elapsed_time = utils.humanize_time(end_normals - start_normals)
    print("Time adjusting normals was: {}".format(elapsed_time))
    # Create output image vector
    w, h, _ = normals.shape
    return normals, w, h


def use_simple_shading(normals, w, h):
    print("Shading using a simple shader...")
    output = np.zeros((h, w), dtype=np.uint8)
    for i in range(w):
        for j in range(h):
            n = normals[j][i]
            output[j][i] = shade(n, L)
    return output


def use_images(normals, w, h):
    print("Opening dark image...")
    dark_img = Image.open(DARK_IMG_FILENAME)
    print("Opening light image...")
    light_img = Image.open(LIGHT_IMG_FILENAME)
    dark_array = np.asarray(dark_img)
    light_array = np.asarray(light_img)
    output = np.zeros((h, w, RGB_CHANNELS), dtype=np.uint8)
    print("Shading between light and dark images...")
    for i in range(w):
        for j in range(h):
            n = normals[j][i]
            dark = dark_array[j][i]
            light = light_array[j][i]
            output[j][i] = shade_colors(n, L, dark, light)
    return output


def use_colors(normals, w, h):
    output = np.zeros((w, h, RGB_CHANNELS), dtype=np.uint8)
    print("Shading between light and dark colors...")
    for i in range(w):
        for j in range(h):
            n = normals[j][i]
            output[j][i] = shade_colors(n, L, DARK_COLOR, LIGHT_COLOR)
    return output


def use_specular_with_images(normals, w, h, ks):
    print("Opening dark image...")
    dark_img = Image.open(DARK_IMG_FILENAME)
    print("Opening light image...")
    light_img = Image.open(LIGHT_IMG_FILENAME)
    dark_array = np.asarray(dark_img)
    light_array = np.asarray(light_img)
    output = np.zeros((h, w, RGB_CHANNELS), dtype=np.uint8)
    print("Shading between light and dark images...")
    for i in range(w):
        for j in range(h):
            n = normals[j][i]
            dark = dark_array[j][i]
            light = light_array[j][i]
            output[j][i] = shade_with_specular(n, L, dark, light, ks)
    return output


def main():
    start = time.time()
    normals_opt = input(
        "Enter [1] to use normal map from image or [2] for using a function\n"
    )
    normals_opt = str(normals_opt)
    # normals_opt = '2'
    if normals_opt == '2':
        if os.path.exists(CREATED_NORMALS_IMG_FILENAME):
            # Load normals from file
            print(
                "Opening previously created normal map {}".format(
                    CREATED_NORMALS_FILENAME
                )
            )
            normal_img = Image.open(CREATED_NORMALS_IMG_FILENAME)
        else:
            normal_img = create_normal_map()
    else:
        print("Opening Normal Map...")
        normal_img = Image.open(NORMAL_MAP_FILENAME)
    # Create a normal vector field from an image map
    normals, w, h = use_normal_map(normal_img, normals_opt)
    # Start shading using the normals
    shading_opt = input(
        "Enter a shading option:\n" \
        "[1] grayscale\n" \
        "[2] 2 colors\n" \
        "[3] diffuse\n" \
        "[4] diffuse + specular\n"
    )
    # shading_opt = '1'
    if shading_opt == '1':
        output = use_simple_shading(normals, w, h)
    elif shading_opt == '2':
        output = use_colors(normals, w, h)
    elif shading_opt == '3':
        output = use_images(normals, w, h)
    else:
        ks = float(input("Enter a size for specular\n"))
        output = use_specular_with_images(normals, w, h, ks)
    # Turn output into image and show it
    im_output = Image.fromarray(output)
    im_output.save(OUTPUT_FILENAME)
    print("Output image saved as {}".format(OUTPUT_FILENAME))
    end = time.time()
    elapsed_time = (end - start)
    print("Elapsed time was: {}ms".format(elapsed_time * SECOND_TO_MS))
    print("or in human time: {}".format(utils.humanize_time(elapsed_time)))


if __name__ == '__main__':
    main()
