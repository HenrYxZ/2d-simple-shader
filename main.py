import time
from PIL import Image
from PIL import ImageFilter
import numpy as np
import os.path
# Local libraries
import utils
import shaders
import logs
# Constants
from constants import MAX_COLOR_VALUE

NORMAL_MAP_FILENAME = "normal.jpg"
HEIGHT_MAP_FILENAME = "height_map.jpg"
OUTPUT_FILENAME = "img_out_"
NORMAL_VECTORS_FILENAME = "normals"
NORMAL_VECTORS_FILE_EXT = ".npy"
SECOND_TO_MS = 1000
NORMAL_DIMENSIONS = 3
DARK_IMG_FILENAME = "dark.jpg"
LIGHT_IMG_FILENAME = "light.jpg"
ENV_IMAGE_FILENAME = "env.jpg"
# BACKGROUND_IMAGE_FILENAME = "background.jpg"
BACKGROUND_IMAGE_FILENAME = "checkers.png"
RGB_CHANNELS = 3
DEFAULT_NORMALS_SIZE = 512
DEFAULT_HEIGHT_MAP_SIZE = 512
CREATED_NORMALS_IMG_FILENAME = "created_normals.jpg"
CREATED_HEIGHT_MAP_FILENAME = "created_height_map.jpg"
CREATED_HEIGHT_MAP_ARRAY_FILENAME = "created_height_map.npy"
DEFAULT_IMG_FORMAT = ".jpg"
LIGHT_COLOR = np.array([232, 158, 39])
DARK_COLOR = np.array([14, 5, 74])
BEST_JPEG_QUALITY = 95
HEIGHT_MAP_CONE_RADIUS = 200
DEFAULT_SPECULAR_SIZE = 0.8
# The program will output an update every (this number) percent done
PERCENTAGE_STEP = 10
L = utils.normalize(np.array([1, 1, 1]))


# Handling Normals ------------------------------------------------------------

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
    h, w, _ = rgb_normal_map.shape
    iterations = w * h
    step_size = np.ceil((iterations * PERCENTAGE_STEP) / 100).astype('int')
    normals = np.zeros((h, w, NORMAL_DIMENSIONS))
    counter = 0
    for i in range(w):
        for j in range(h):
            if counter % step_size == 0:
                percent_done = int((counter / float(iterations)) * 100)
                print("{}% of normal vectors created".format(percent_done))
            normals[j][i] = utils.adjust(rgb_normal_map[j][i][:3])
            counter += 1
    return normals


def inverse_adjust_normal_map(normals):
    """
    Return a RGB normal map created from an array of normalized vector normals.
    Args:
        normals: Normalized vector normals

    Returns:
        Image: RGB image for the normal map corresponding to this normals
    """
    print("Creating RGB Normal Map image from normals...")
    h, w, channels = normals.shape
    rgb_array = np.zeros((h, w, channels), dtype=np.uint8)
    for j in range(h):
        for i in range(w):
            r = (2 * normals[j][i][0] - 1) * MAX_COLOR_VALUE
            g = (2 * normals[j][i][1] - 1) * MAX_COLOR_VALUE
            b = normals[j][i][2] * MAX_COLOR_VALUE
            rgb_array[j][i] = (r, g, b)
    return Image.fromarray(rgb_array)


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
    normals_img.save(CREATED_NORMALS_IMG_FILENAME, quality=BEST_JPEG_QUALITY)
    return normals_img


def use_normal_map(normal_img, normal_opt):
    # Create an array from the image
    normal_im_array = np.asarray(normal_img)
    # TODO move this up
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


def create_height_map():
    """
    Create a height map image from a function, save it and return it.

    Returns:
        Image: The height map in 'L' mode of PIL Image
    """
    print("Creating Height map from a function...")
    cone_h = MAX_COLOR_VALUE
    cone_r = HEIGHT_MAP_CONE_RADIUS
    h = DEFAULT_HEIGHT_MAP_SIZE
    w = DEFAULT_HEIGHT_MAP_SIZE
    output = np.zeros((h, w), dtype=np.uint8)
    for j in range(DEFAULT_HEIGHT_MAP_SIZE):
        for i in range(DEFAULT_HEIGHT_MAP_SIZE):
            x = i - w / 2
            y = j - h / 2
            z = (cone_h / cone_r) * (max(0, cone_r - np.sqrt(x**2 + y**2)))
            output[j][i] = z
    height_map = Image.fromarray(output)
    height_map.save(CREATED_HEIGHT_MAP_FILENAME, quality=BEST_JPEG_QUALITY)
    return height_map


def use_height_map(height_map, normals_opt):
    """
    Use the height map to return an array of normals.
    Args:
        height_map(Image): The height map image
        normals_opt(char): The option used for getting normals
    Returns:
        np.array: a matrix array of unit normal vectors for the map
        int: the width of the normals array
        int: the height of the normals array
    """
    print("Using height map...")
    # TODO move this up
    normals_filename = (
            NORMAL_VECTORS_FILENAME + normals_opt + NORMAL_VECTORS_FILE_EXT
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
    # dx_kernel = (-1, 0, 1, -2, 0, 2, -1, 0, 1)
    # dy_kernel = (-1, -2, -1, 0, 0, 0, 1, 2, 1)
    # kernel_size = (3, 3)
    # dx_img = height_map.filter(
    #     ImageFilter.Kernel(kernel_size, kernel=dx_kernel)
    # )
    # dy_img = height_map.filter(
    #     ImageFilter.Kernel(kernel_size, kernel=dy_kernel)
    # )
    # dx_arr = np.asarray(dx_img)
    # dy_arr = np.asarray(dy_img)
    w, h = height_map.size
    height_map_arr = np.asarray(height_map)
    normals = np.zeros((h, w, NORMAL_DIMENSIONS))
    for j in range(h):
        for i in range(w):
            # dx = dx_arr[j][i] / MAX_COLOR_VALUE
            # dy = dy_arr[j][i] / MAX_COLOR_VALUE
            if i > 1 and i < (w - 2):
                x1 = float(height_map_arr[j][i + 1])
                x0 = float(height_map_arr[j][i - 1])
                dx = (x1 - x0) / (MAX_COLOR_VALUE)
                # dx = (x1 - x0) / 2.0
            else:
                dx = 0.0
            if j > 1 and j < (h - 2):
                y0 = 2 * float(height_map_arr[j + 1][i])
                y1 = 2 * float(height_map_arr[j - 1][i])
                dy = (y1 - y0) / (MAX_COLOR_VALUE)
                # dy = (y1 - y0) / 2.0
            else:
                dy = 0.0
            n = np.array([dx, dy, 1.0])
            normals[j][i] = utils.normalize(n)
    # Only for debugging purposes save an image
    normals_img = inverse_adjust_normal_map(normals)
    img_filename = "from_height_map_{}.jpg".format(normals_opt)
    normals_img.save(img_filename, quality=BEST_JPEG_QUALITY)
    # np.save(normals_filename, normals)
    return normals, w, h
# -----------------------------------------------------------------------------


def use_simple_shading(normals, w, h):
    print("Shading using a simple shader...")
    output = np.zeros((h, w), dtype=np.uint8)
    for i in range(w):
        for j in range(h):
            n = normals[j][i]
            output[j][i] = shaders.shade(n, L)
    return output


def use_reflection(normals, w, h, kr):
    env_img = Image.open(ENV_IMAGE_FILENAME)
    env_arr = np.asarray(env_img)
    print("Shading using reflection...")
    output = np.zeros((h, w, RGB_CHANNELS), dtype=np.uint8)
    for i in range(w):
        for j in range(h):
            n = normals[j][i]
            output[j][i] = shaders.shade_reflection(n, L, kr, i, j, env_arr)
    return output


def use_refraction(normals, w, h, kr, ior):
    background_img = Image.open(BACKGROUND_IMAGE_FILENAME)
    background_arr = np.asarray(background_img)
    print("Shading using refraction...")
    output = np.zeros((h, w, RGB_CHANNELS), dtype=np.uint8)
    counter = 0
    step_counter = 1
    step_size = np.ceil((w * h * PERCENTAGE_STEP) / 100).astype('int')
    for i in range(w):
        for j in range(h):
            n = normals[j][i]
            output[j][i] = shaders.shade_refraction(
                n, L, kr, ior, i, j, background_arr
            )
            if counter % step_size == 0 and counter > 0:
                print("{}%".format(step_counter * PERCENTAGE_STEP))
                step_counter += 1
            counter += 1
    return output


def use_fresnel(normals, w, h, kr, ior):
    env_img = Image.open(ENV_IMAGE_FILENAME)
    env_arr = np.asarray(env_img)
    background_img = Image.open(BACKGROUND_IMAGE_FILENAME)
    background_arr = np.asarray(background_img)
    print("Shading using fresnel...")
    output = np.zeros((h, w, RGB_CHANNELS), dtype=np.uint8)
    counter = 0
    step_counter = 1
    step_size = np.ceil((w * h * PERCENTAGE_STEP) / 100).astype('int')
    for i in range(w):
        for j in range(h):
            n = normals[j][i]
            output[j][i] = shaders.shade_fresnel(
                n, L, kr, ior, i, j, env_arr, background_arr
            )
            if counter % step_size == 0 and counter > 0:
                print("{}%".format(step_counter * PERCENTAGE_STEP))
                step_counter += 1
            counter += 1
    return output


def use_colors(normals, w, h):
    output = np.zeros((w, h, RGB_CHANNELS), dtype=np.uint8)
    print("Shading between light and dark colors...")
    for i in range(w):
        for j in range(h):
            n = normals[j][i]
            output[j][i] = shaders.shade_lambert(n, L, DARK_COLOR, LIGHT_COLOR)
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
        "Enter an option to get the normals:\n"
        "[1] normal map from image\n"
        "[2] using a function\n"
        "[3] height map from image\n"
        "[4] using a function for height map\n"
    )
    normals_opt = str(normals_opt)
    if normals_opt == '2':
        if os.path.exists(CREATED_NORMALS_IMG_FILENAME):
            print(
                "Opening previously created normal map {}".format(
                    CREATED_NORMALS_IMG_FILENAME
                )
            )
            normal_img = Image.open(CREATED_NORMALS_IMG_FILENAME)
        else:
            normal_img = create_normal_map()
        normals, w, h = use_normal_map(normal_img, normals_opt)
    elif normals_opt == '3':
        print("Opening Height Map...")
        height_map = Image.open(HEIGHT_MAP_FILENAME)
        r, g, b = height_map.split()
        height_map = r
        normals, w, h = use_height_map(height_map, normals_opt)
    elif normals_opt == '4':
        if os.path.exists(CREATED_HEIGHT_MAP_FILENAME):
            print(
                "Opening previously created height map {}".format(
                    CREATED_HEIGHT_MAP_FILENAME
                )
            )
            height_map = Image.open(CREATED_HEIGHT_MAP_FILENAME)
        else:
            height_map = create_height_map()
        normals, w, h = use_height_map(height_map, normals_opt)
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
            "[7] refraction\n"
            "[8] fresnel\n"
        )
    )
    if shading_opt == '1':
        output = use_simple_shading(normals, w, h)
    elif shading_opt == '2':
        output = use_colors(normals, w, h)
    elif shading_opt == '3':
        output = shade_with_images(
            normals, w, h, shaders.shade_lambert, logs.SHADING_IMAGES
        )
    elif shading_opt == '4':
        # ks = float(input("Enter a size for specular\n"))
        ks = DEFAULT_SPECULAR_SIZE
        output = shade_with_images(
            normals, w, h, shaders.shade_with_specular, logs.SHADING_IMAGES, ks
        )
    elif shading_opt == '5':
        # ks = float(input("Enter a size for specular\n"))
        ks = DEFAULT_SPECULAR_SIZE
        thickness = float(
            input("Enter a thickness for border (float between 0 and 1)\n")
        )
        output = shade_with_images(
            normals, w, h, shaders.shade_specular_border, logs.SHADING_IMAGES,
            ks, thickness
        )
    elif shading_opt == '6':
        kr = 0.25
        output = use_reflection(normals, w, h, kr)
    elif shading_opt == '7':
        kr = 0.25
        # ior = float(input("Enter Index of Refraction\n"))
        ior = 0.66
        output = use_refraction(normals, w, h, kr, ior)
    else:
        kr = 0.25
        ior = 0.66
        output = use_fresnel(normals, w, h, kr, ior)
    # Turn output into image and show it
    im_output = Image.fromarray(output)
    output_img_filename = (
        OUTPUT_FILENAME + normals_opt + str(shading_opt) + DEFAULT_IMG_FORMAT
    )
    im_output.save(output_img_filename, quality=BEST_JPEG_QUALITY)
    print("Output image saved as {}".format(output_img_filename))
    end = time.time()
    elapsed_time = (end - start)
    print("Elapsed time was: {}ms".format(elapsed_time * SECOND_TO_MS))
    print("or in human time: {}".format(utils.humanize_time(elapsed_time)))


if __name__ == '__main__':
    main()
