import numpy as np
import os.path
from progress.bar import Bar
import utils


NORMAL_VECTORS_FILENAME = "normals.npy"
NORMAL_DIMENSIONS = 3


def create_normals(normal_map):
    h, w, _ = normal_map.shape
    iterations = w * h
    step_size = np.ceil(iterations / 100).astype(int)
    normals = np.zeros((h, w, NORMAL_DIMENSIONS))
    counter = 0
    bar = Bar("Processing Normals...", max=100, suffix='%(percent)d%%')
    bar.check_tty = False
    for i in range(w):
        for j in range(h):
            normals[j][i] = utils.adjust(normal_map[j][i][:3])
            counter += 1
            if counter % step_size == 0:
                bar.next()
    bar.finish()
    np.save(NORMAL_VECTORS_FILENAME, normals)
    print(f"Normal vectors stored in {NORMAL_VECTORS_FILENAME}")
    return normals


def get_normals(normal_map):
    if os.path.exists(NORMAL_VECTORS_FILENAME):
        # Load normals from file
        print(f"Loading normal vectors from file {NORMAL_VECTORS_FILENAME}")
        normals = np.load(NORMAL_VECTORS_FILENAME)
        w, h, _ = normals.shape
        return normals
    # Create the normals vector map
    normals = create_normals(normal_map)
    return normals
