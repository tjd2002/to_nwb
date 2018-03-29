import os

from datetime import datetime
import numpy as np
from scipy.io import loadmat
import h5py


from pynwb import NWBFile, NWBHDF5IO
from pynwb.ecephys import ElectricalSeries, LFP
from pynwb.ophys import OpticalChannel, TwoPhotonSeries
from pynwb.image import ImageSeries


from neuroscope import get_channel_groups, get_lfp_sampling_rate
from general import gzip

NA = 'THIS REQUIRED ATTRIBUTE INTENTIONALLY LEFT BLANK.'


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

lfp_xml_fpath = os.path.join(fpath, 'LFP', 'svr009_Day2_FOV1_170504_131823.xml')
channel_groups = get_channel_groups(lfp_xml_fpath)
lfp_channels = channel_groups[0]
lfp_fs = get_lfp_sampling_rate(lfp_xml_fpath)
nchannels = sum(len(x) for x in channel_groups)

eeg_file = os.path.join(fpath, 'LFP/svr009_Day2_FOV1_170504_131823.eeg')
all_channels = np.fromfile(eeg_file, dtype=np.int16).reshape(-1, nchannels)
lfp_signal = all_channels[:, lfp_channels]

device_name = 'LFP device'
device = nwbfile.create_device(device_name, source=source)
electrode_group = nwbfile.create_electrode_group(
    name=device_name + '_electrodes',
    source=fname + '.xml',
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
    imaging_data = f['imaging'][:]

image_series = TwoPhotonSeries(name='test_iS', source='Ca2+ imaging example', dimension=[2],
                               data=imaging_data, imaging_plane=imaging_plane,
                               starting_frame=[0], timestamps=list())
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

f.close()


