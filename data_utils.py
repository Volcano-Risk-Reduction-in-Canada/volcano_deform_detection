
import numpy as np


def get_percent_above_50_80(probMap):
    total_pixels = probMap.size
    above_50_percent = np.sum(probMap > 0.5)
    above_80_percent = np.sum(probMap > 0.8)

    percent_above_50 = (above_50_percent / total_pixels) * 100
    percent_above_80 = (above_80_percent / total_pixels) * 100
    return percent_above_50, percent_above_80