import numpy as np
from progress.bar import Bar

# Local Modules
from constants import COLOR_CHANNELS, MAX_COLOR_VALUE
import shaders
import utils


def render_lambert(ambient_map, diffuse_map, normals, light_pos):
    h, w, _ = ambient_map.shape
    bar, step_size = utils.progress_bar(h * w, "Rendering...")
    counter = 0
    output = np.zeros([h, w, COLOR_CHANNELS], dtype=np.uint8)
    for j in range(h):
        for i in range(w):
            ambient = ambient_map[j][i]
            diffuse = diffuse_map[j][i]
            n = normals[j][i]
            l = utils.normalize(light_pos - np.array([i, h - j, 0]))
            color = shaders.shade_lambert(n, l, ambient, diffuse)
            output[j][i] = np.round(color * MAX_COLOR_VALUE)
            counter += 1
            if counter % step_size == 0:
                bar.next()
    bar.finish()
    return output


def render_specular(ambient_map, diffuse_map, specular_map, normals, light_pos):
    h, w, _ = ambient_map.shape
    bar, step_size = utils.progress_bar(h * w, "Rendering...")
    counter = 0
    output = np.zeros([h, w, COLOR_CHANNELS], dtype=np.uint8)
    for j in range(h):
        for i in range(w):
            ambient = ambient_map[j][i]
            diffuse = diffuse_map[j][i]
            specular = specular_map[j][i]
            n = normals[j][i]
            l = utils.normalize(light_pos - np.array([i, h - j, 0]))
            color = shaders.shade_with_specular(
                n, l, ambient, diffuse, specular
            )
            output[j][i] = np.round(color * MAX_COLOR_VALUE)
            counter += 1
            if counter % step_size == 0:
                bar.next()
    bar.finish()
    return output
