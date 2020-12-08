import imageio
import numpy as np
import os
import os.path
from PIL import Image

# Local Modules
from constants import MAX_COLOR_VALUE, MAX_QUALITY
from normal_map import get_normals
from render import render_lambert, render_specular
import utils

AMBIENT_FILENAME = "dark.jpg"
DIFFUSE_FILENAME = "light.jpg"
FPS = 24
NORMAL_FILENAME = "normal_map_sm.png"
SPECULAR_FILENAME = "specular_sm.jpg"
Z_HEIGHT = 512
OUTPUT_DIR = "output"
NUM_FRAMES = 48

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
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)


def main():
    global light_pos
    init()
    while True:
        opt = input(
            "Enter an option:\n"
            "[1] for 2D Lambert\n"
            "[2] for 2D Specular\n"
            "[3] for 2D Reflection\n"
            "[4] for 2D Refraction\n"
            "[5] for 2D Fresnel\n"
            "[0] to quit\n"
        )
        if opt == '0':
            quit()
        timer = utils.Timer()
        timer.start()
        if opt == '1':
            print("Using 2D Lambert...")
            opt = input(
                "Enter an option:\n"
                "[1] for single image\n"
                "[2] for video\n"
            )
            if opt == '1':
                # Single image
                output = render_lambert(
                    ambient_map, diffuse_map, normals, light_pos
                )
                output_img = Image.fromarray(output)
                output_img.save("lambert.jpg", quality=MAX_QUALITY)
                print("Image saved in lambert.jpg")
            else:
                # Video
                writer = imageio.get_writer(
                    f"{OUTPUT_DIR}/lambert.mp4", format="mp4", mode='I', fps=FPS
                )
                for i in range(NUM_FRAMES):
                    angle = i * (2 * np.pi / (NUM_FRAMES - 1))
                    # Assume we work in a cube with size Z_HEIGHT
                    r = Z_HEIGHT // 2
                    center = (r, r)
                    x = r * np.cos(angle) + center[0]
                    y = r * np.sin(angle) + center[1]
                    light_pos = np.array([x, y, Z_HEIGHT])
                    output = render_lambert(
                        ambient_map, diffuse_map, normals, light_pos
                    )
                    writer.append_data(output)
                    print(f"Image n° {i + 1}/{NUM_FRAMES} done")
                writer.close()
        elif opt == '2':
            specular_map = open_image(SPECULAR_FILENAME)
            opt = input(
                "Enter an option:\n"
                "[1] for single image\n"
                "[2] for video\n"
            )
            if opt == '1':
                # Single image
                output = render_specular(
                    ambient_map, diffuse_map, specular_map, normals, light_pos
                )
                output_img = Image.fromarray(output)
                output_img.save("specular_out.jpg", quality=MAX_QUALITY)
                print("Image saved in specular_out.jpg")
            else:
                # Video
                writer = imageio.get_writer(
                    f"{OUTPUT_DIR}/specular.mp4", format="mp4", mode='I',
                    fps=FPS
                )
                for i in range(NUM_FRAMES):
                    angle = i * (2 * np.pi / (NUM_FRAMES - 1))
                    # Assume we work in a cube with size Z_HEIGHT
                    r = Z_HEIGHT // 2
                    center = (r, r)
                    x = r * np.cos(angle) + center[0]
                    y = r * np.sin(angle) + center[1]
                    light_pos = np.array([x, y, Z_HEIGHT])
                    output = render_specular(
                        ambient_map, diffuse_map, specular_map, normals,
                        light_pos
                    )
                    writer.append_data(output)
                    print(f"Image n° {i + 1}/{NUM_FRAMES} done")
                writer.close()

        timer.stop()
        print(f"Total time spent: {timer}")


if __name__ == '__main__':
    main()
