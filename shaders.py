import numpy as np
# local modules
import utils
from constants import MAX_COLOR_VALUE, COLOR_FOR_LIGHT, COLOR_FOR_BORDER

DISTANCE_TO_ENV_MAP = 10


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
    Shader calculation for normal and light vectors, dark and light colors and
    specular size ks.
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
    s = l[2] * -1 + 2 * n[2] * n_dot_l
    s = np.maximum(0, s)
    # try smoothstep
    min = 0.01
    max = 0.99
    if s < min:
        s = 0
    elif s > max:
        s = 1
    else:
        s = -2 * (s ** 3) + 3 * (s ** 2)
    alpha = 2
    s = s ** alpha
    color = color * (1 - s * ks) + s * ks * COLOR_FOR_LIGHT
    return color


def shade_specular_border(n, l, dark, light, ks, thickness):
    """
    Shader calculation for normal and light vectors, dark and light colors,
    and ks specular size and thickness of border parameters.
    Args:
        n(numpy.array): Unit normal vector
        l(numpy.array): Unit vector in the direction to the light
        dark(numpy.array): RGB dark color
        light(numpy.array): RGB light color
        ks(float): size of specularity (this can be changed by the user)
        thickness(float): thickness parameter for the border defined by user

    Returns:
        numpy.uint8: The calculated color (RGB)
    """

    eye = np.array([0, 0, 1])
    b = np.maximum(0, 1 - np.dot(eye, n))
    min = thickness
    max = 1
    b = (b - min) / (max - min)
    if b < min:
        b = 0
    elif b > max:
        b = 1
    color = shade_with_specular(n, l, dark, light, ks)
    color = color * (1 - b) + b * COLOR_FOR_BORDER
    return color


def shade_reflection(n, l, kr, i, j, env_arr):
    """
    Shader calculation for a normal and a light vector.
    Args:
        n(numpy.array): Unit normal vector
        l(numpy.array): Unit vector in the direction to the light
        kr(float): Coefficient of reflection [0..1]
        i(int): Position of this pixel on x
        j(int): Position of this pixel on y
        env_arr(numpy.array): Array for the environment map
    Returns:
        numpy.uint8: The calculated color (RGB)
    """
    greyscale_color = shade(n, l)
    color = np.array([greyscale_color, greyscale_color, greyscale_color])
    a, b, c = n
    h, w = env_arr.shape[:2]
    i_prime = (2 * a * c * DISTANCE_TO_ENV_MAP) / (-1 + 2 * (c ** 2)) + i
    j_prime = (2 * b * c * DISTANCE_TO_ENV_MAP) / (-1 + 2 * (c ** 2)) + j
    i_prime = int(round(i_prime)) % w
    j_prime = int(round(j_prime)) % h
    reflected_color = env_arr[j_prime][i_prime]
    color = (1 - kr) * color + kr * reflected_color
    return color


def shade_refraction(n, l, kr, ior, i, j, background_arr):
    """
    Shade including refraction
    Args:
        n(numpy.array): Unit normal vector
        l(numpy.array): Unit vector in the direction to the light
        kr(float): Coefficient of reflection [0..1]
        ior(float): Index of Refraction for this interface
        i(int): Position of this pixel on x
        j(int): Position of this pixel on y
        background_arr(numpy.array): Array for the background map

    Returns:
        numpy.uint8: The calculated color (RGB)
    """
    greyscale_color = shade(n, l)
    color = np.array([greyscale_color, greyscale_color, greyscale_color])
    h, w = background_arr.shape[:2]
    v = np.array([0, 0, 1])
    cos_theta_1 = np.dot(v, n)
    term = 1 - (1 - cos_theta_1 ** 2) * (ior ** 2)
    # If term is negative there is no refraction
    if term < 0:
        return color
    T = -1 * v / ior + ((cos_theta_1 / ior) - np.sqrt(term)) * n
    T = utils.normalize(T)
    aT, bT, cT = T
    # This is just in case you want to try different values of d, it's actually
    # a trick to do this, since d would be different for each shading point
    # if cT is different for each shading point
    d = 120
    i_prime = (int(round(aT * (d / cT))) + i) % w
    j_prime = (int(round(bT * (d / cT))) + j) % h
    refracted_color = background_arr[j_prime][i_prime]
    color = (1 - kr) * color + kr * refracted_color
    return color
