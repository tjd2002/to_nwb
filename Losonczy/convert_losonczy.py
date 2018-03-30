import os

from datetime import datetime
import numpy as np
from scipy.io import loadmat
import h5py


from pynwb import NWBFile, NWBHDF5IO
from pynwb.ecephys import ElectricalSeries, LFP
from pynwb.ophys import OpticalChannel, TwoPhotonSeries
from pynwb.image import ImageSeries


from ..neuroscope import get_channel_groups
from ..general import gzip

from .lfp_helpers import loadEEG

NA = 'THIS REQUIRED ATTRIBUTE INTENTIONALLY LEFT BLANK.'
SHORTEN = True


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
all_ts = []

eeg_base_name = os.path.join(fpath, 'LFP', 'svr009_Day2_FOV1_170504_131823')
eeg_dict = loadEEG(eeg_base_name)

lfp_xml_fpath = eeg_base_name + '.xml'
channel_groups = get_channel_groups(lfp_xml_fpath)
lfp_channels = channel_groups[0]
lfp_fs = eeg_dict['sampleFreq']
nchannels = eeg_dict['nChannels']


lfp_signal = eeg_dict['EEG'][:, lfp_channels]

device_name = 'LFP device'
device = nwbfile.create_device(device_name, source=source)
electrode_group = nwbfile.create_electrode_group(
    name=device_name + '_electrodes',
    source=lfp_xml_fpath,
    description=device_name,
    device=device,
    location='unknown')

for channel in channel_groups[0]:
    nwbfile.add_electrode(channel,
                          np.nan, np.nan, np.nan,  # position?
                          imp=np.nan,
                          location='unknown',
                          filtering='unknown',
                          description='lfp electrode {}'.format(channel),
                          group=electrode_group)


lfp_table_region = nwbfile.create_electrode_table_region(list(range(4)),
                                                         'lfp electrodes')

lfp_elec_series = ElectricalSeries('lfp', 'lfp',
                                   gzip(lfp_signal),
                                   lfp_table_region,
                                   conversion=np.nan,
                                   starting_time=0.0,
                                   rate=lfp_fs,
                                   resolution=np.nan)

nwbfile.add_acquisition(LFP(source=source, electrical_series=lfp_elec_series))




optical_channel = OpticalChannel(
    name='Optical Channel',
    source=NA,
    description=NA,
    emission_lambda=NA,
)

imaging_plane = nwbfile.create_imaging_plane('my_imgpln',
                                             'Ca2+ imaging example',
                                             optical_channel,
                                             'a very interesting part of the brain',
                                             'imaging_device_1',
                                             '6.28', '2.718', 'GFP', 'my favorite brain location',
                                             (1, 2, 1, 2, 3), 4.0, 'manifold unit', 'A frame to refer to')

imaging_h5_filepath = '/Users/bendichter/Desktop/Losonczy/from_sebi/example_data/TSeries-05042017-001_Cycle00001_Element00001.h5'


with h5py.File(imaging_h5_filepath, 'r') as f:
    if SHORTEN:
        imaging_data = f['imaging'][:100, ...]
    else:
        imaging_data = f['imaging'][:]

image_series = TwoPhotonSeries(name='image', source='Ca2+ imaging example', dimension=[2],
                               data=imaging_data, imaging_plane=imaging_plane,
                               starting_frame=[0], timestamps=[1,2,3], scan_line_rate=np.nan,
                               pmt_gain=np.nan)
nwbfile.add_acquisition(image_series)




out_fname = 'sebi_data.nwb'
print('writing NWB file...', end='', flush=True)
with NWBHDF5IO(out_fname, mode='w') as io:
    io.write(nwbfile, cache_spec=False)
print('done.')

print('testing read...', end='', flush=True)
# test read
with NWBHDF5IO(out_fname, mode='r') as io:
    io.read()
print('done.')



