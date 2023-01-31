import pyBigWig
import numpy as np


class Conservation:
    def __init__(self, bw_location):
        self.bw = pyBigWig.open(str(bw_location))

    def get_values(self, chr, start, end):
        return np.array(self.bw.values(chr, start, end))
