import numpy as np


def round_array_to_nearest_N(array, N):
    return N * np.round(array / N).astype(int)