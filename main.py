import time
from PIL import Image
import numpy as np
import os.path
# Local libraries
import utils
import shaders
import logs
# Constants
from constants import MAX_COLOR_VALUE

NORMAL_MAP_FILENAME = "normal.jpg"
OUTPUT_FILENAME = "img_out_"
NORMAL_VECTORS_FILENAME = "normals"
NORMAL_VECTORS_FILE_EXT = ".npy"
SECOND_TO_MS = 1000
NORMAL_DIMENSIONS = 3
DARK_IMG_FILENAME = "dark.jpg"
LIGHT_IMG_FILENAME = "light.jpg"
ENV_IMAGE_FILENAME = "env.jpg"
RGB_CHANNELS = 3
DEFAULT_NORMALS_SIZE = 512
CREATED_NORMALS_FILENAME = "created_normals.npy"
CREATED_NORMALS_IMG_FILENAME = "created_normals.jpg"
DEFAULT_IMG_FORMAT = ".jpg"
LIGHT_COLOR = np.array([232, 158, 39])
DARK_COLOR = np.array([14, 5, 74])
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
    # TODO: Maybe change this
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
    normals_img.save(CREATED_NORMALS_IMG_FILENAME)
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
            output[j][i] = shaders.shade(n, L)
    return output


def use_reflection(normals, w, h):
    env_img = Image.open(ENV_IMAGE_FILENAME)
    env_arr = np.asarray(env_img)
    print("Shading using a reflection...")
    output = np.zeros((h, w, RGB_CHANNELS), dtype=np.uint8)
    for i in range(w):
        for j in range(h):
            n = normals[j][i]
            output[j][i] = shaders.shade_reflection(n, L, i, j, env_arr)
    return output


def use_colors(normals, w, h):
    output = np.zeros((w, h, RGB_CHANNELS), dtype=np.uint8)
    print("Shading between light and dark colors...")
    for i in range(w):
        for j in range(h):
            n = normals[j][i]
            output[j][i] = shaders.shade_colors(n, L, DARK_COLOR, LIGHT_COLOR)
    return output


def shade_with_images(normals, w, h, shading_function, shading_str, *args):
    print("Opening dark image...")
    dark_img = Image.open(DARK_IMG_FILENAME)
    print("Opening light image...")
    light_img = Image.open(LIGHT_IMG_FILENAME)
    dark_array = np.asarray(dark_img)
    light_array = np.asarray(light_img)
    output = np.zeros((h, w, RGB_CHANNELS), dtype=np.uint8)
    print(shading_str)
    for i in range(w):
        for j in range(h):
            n = normals[j][i]
            dark = dark_array[j][i]
            light = light_array[j][i]
            output[j][i] = shading_function(n, L, dark, light, *args)
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
    shading_opt = str(
        input(
            "Enter a shading option:\n"
            "[1] grayscale\n"
            "[2] 2 colors\n"
            "[3] diffuse\n"
            "[4] diffuse + specular\n"
            "[5] diffuse + specular + border\n"
            "[6] reflection\n"
        )
    )
    if shading_opt == '1':
        output = use_simple_shading(normals, w, h)
    elif shading_opt == '2':
        output = use_colors(normals, w, h)
    elif shading_opt == '3':
        output = shade_with_images(
            normals, w, h, shaders.shade_colors, logs.SHADING_IMAGES
        )
    elif shading_opt == '4':
        ks = float(input("Enter a size for specular\n"))
        output = shade_with_images(
            normals, w, h, shaders.shade_with_specular, logs.SHADING_IMAGES, ks
        )
    elif shading_opt == '5':
        ks = float(input("Enter a size for specular\n"))
        thickness = float(
            input("Enter a thickness for border (float between 0 and 1)\n")
        )
        output = shade_with_images(
            normals, w, h, shaders.shade_specular_border, logs.SHADING_IMAGES,
            ks, thickness
        )
    else:
        output = use_reflection(normals, w, h)
    # Turn output into image and show it
    im_output = Image.fromarray(output)
    output_img_filename = (
        OUTPUT_FILENAME + normals_opt + str(shading_opt) + DEFAULT_IMG_FORMAT
    )
    im_output.save(output_img_filename)
    print("Output image saved as {}".format(output_img_filename))
    end = time.time()
    elapsed_time = (end - start)
    print("Elapsed time was: {}ms".format(elapsed_time * SECOND_TO_MS))
    print("or in human time: {}".format(utils.humanize_time(elapsed_time)))


if __name__ == '__main__':
    main()
