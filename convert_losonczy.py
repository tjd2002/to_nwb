import os

from datetime import datetime
import numpy as np
from scipy.io import loadmat
import pdb


from pynwb import NWBFile, NWBHDF5IO

from neuroscope import get_channel_groups

fpath = '/Users/bendichter/Desktop/Losonczy/from_sebi/example_data'
fpath_base, fname = os.path.split(fpath)
session_description = 'Example data from Sebi'
identifier = fname
session_start_time = datetime(2017, 5, 4)
institution = 'Columbia'
lab = 'Losonczy'


source = fname
nwbfile = NWBFile(source, session_description, identifier,
                  session_start_time, datetime.now(),
                  institution=institution, lab=lab)

channels = get_channel_groups(os.path.join(fpath, 'LFP'),
                              'svr009_Day2_FOV1_170504_131823')
nchannels = sum(len(x) for x in channels)

eeg_file = os.path.join(fpath, 'LFP/svr009_Day2_FOV1_170504_131823.eeg')
all_channels = np.fromfile(eeg_file, dtype=np.int16).reshape(-1, nchannels)

device_name = 'LFP device'
device = nwbfile.create_device(device_name)
electrode_group = nwbfile.create_electrode_group(
    name=device_name + '_electrodes',
    source=fname + '.xml',
    description=device_name,
    device=device,
    location='unknown')

for channel in channels[0]:
    nwbfile.add_electrode(channel,
                          np.nan, np.nan, np.nan,  # position?
                          imp=np.nan,
                          location='unknown',
                          filtering='unknown',
                          description='electrode {}'.format(channel),
                          group=electrode_group)

lfp_table_region = nwbfile.create_electrode_table_region(channels,
                                                         'lfp electrode')



