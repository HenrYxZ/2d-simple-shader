import numpy as np
from PIL import Image

# Local Modules
from constants import MAX_COLOR_VALUE, MAX_QUALITY
from normal_map import get_normals
from render import render_lambert, render_specular
import utils

AMBIENT_FILENAME = "ambient.jpg"
DIFFUSE_FILENAME = "diffuse.jpg"
NORMAL_FILENAME = "normal_map.png"
SPECULAR_FILENAME = "specular.jpg"
Z_HEIGHT = 1024

ambient_map = None
diffuse_map = None
normal_map = None
normals = None
light_pos = None


def open_image(img_filename):
    img = Image.open(img_filename)
    img_arr = np.array(img) / MAX_COLOR_VALUE
    return img_arr


def init():
    global ambient_map, diffuse_map, normal_map, normals, light_pos
    ambient_map = open_image(AMBIENT_FILENAME)
    diffuse_map = open_image(DIFFUSE_FILENAME)
    normal_map = open_image(NORMAL_FILENAME)
    normals = get_normals(normal_map)
    h, w, _ = normal_map.shape
    center = (w // 2, h // 2)
    light_pos = np.array([center[0], center[1], Z_HEIGHT])


def main():
    init()
    while True:
        opt = input(
            "Enter an option:\n"
            "[1] for 2D Lambert \n"
            "[2] for 2D Specular\n"
            "[3] for 2D Refraction\n"
            "[4] for 2D Reflection\n"
            "[0] to quit\n"
        )
        if opt == '0':
            quit()
        timer = utils.Timer()
        timer.start()
        if opt == '1':
            print("Using 2D Lambert...")
            output = render_lambert(
                ambient_map, diffuse_map, normals, light_pos
            )
        else:
            print("Using 2D Specular...")
            specular_map = open_image(SPECULAR_FILENAME)
            output = render_specular(
                ambient_map, diffuse_map, specular_map, normals, light_pos
            )
        output_img = Image.fromarray(output)
        output_img.save("output.jpg", quality=MAX_QUALITY)
        print("Image saved in output.jpg")
        timer.stop()
        print(f"Total time spent: {timer}")


if __name__ == '__main__':
    main()
