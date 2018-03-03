import os
import numpy as np

fpath_base = '/Users/bendichter/Desktop/Schnitzer/data/test1_171207_181558'

amp_fpath = os.path.join(fpath_base, 'amplifier.dat')

amp_data = np.fromfile(amp_fpath, dtype=np.int16).reshape(-1, 32)
