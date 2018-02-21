import os
from datetime import datetime
from functools import partialmethod

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from pynwb import NWBFile, NWBHDF5IO
from pynwb.behavior import SpatialSeries, Position
from pynwb.ecephys import ElectricalSeries, Clustering
from pynwb.form.backends.hdf5 import H5DataIO
from scipy.io import loadmat

from utils import find_discontinuities

gzip = H5DataIO
gzip.__init__ = partialmethod(H5DataIO.__init__, compress=True)


def load_xml(filepath):
    with open(filepath, 'r') as xml_file:
        contents = xml_file.read()
        soup = BeautifulSoup(contents, 'xml')
    return soup


def get_channel_groups(fpath, fname):
    """Get the groups of channels that are recorded on each shank from the xml
    file

    Parameters
    ----------
    fpath: str
    fname: str

    Returns
    -------
    list(list)

    """
    xml_filepath = os.path.join(fpath, fname + '.xml')
    soup = load_xml(xml_filepath)

    channel_groups = [[int(channel.string)
                       for channel in group.find_all('channel')]
                      for group in soup.channelGroups.find_all('group')]

    return channel_groups


def get_shank_channels(fpath, fname):
    """Read the channels on the shanks in Neuroscope xml

    Parameters
    ----------
    fpath
    fname

    Returns
    -------

    """
    xml_filepath = os.path.join(fpath, fname + '.xml')
    soup = load_xml(xml_filepath)

    shank_channels = [[int(channel.string)
                       for channel in group.find_all('channel')]
                      for group in soup.spikeDetection.channelGroups.find_all('group')]
    return shank_channels


def get_lfp_sampling_rate(fpath, fname):
    """Reads the LFP Sampling Rate from the xml parameter file of the
    Neuroscope format

    Parameters
    ----------
    fpath: str
    fname: str

    Returns
    -------
    fs: float

    """
    xml_filepath = os.path.join(fpath, fname + '.xml')
    soup = load_xml(xml_filepath)

    return float(soup.lfpSamplingRate.string)


def get_position_data(fpath, fname, fs=1250./32.,
                      names=('x0', 'y0', 'x1', 'y1')):
    """Read raw position sensor data from .whl file

    Parameters
    ----------
    fpath: str
    fname: str
    fs: float
    names: iterable
        names of column headings

    Returns
    -------
    df: pandas.DataFrame
    """
    df = pd.read_csv(os.path.join(fpath, fname + '.whl'),
                     sep='\t', names=names)

    df.index = np.arange(len(df)) / fs
    df.index.name = 'tt (sec)'

    return df


def get_clusters_single_shank(fpath, fname, shankn):
    """Read the spike time data for a from the .res and .clu files for a single
    shank. Automatically removes noise and multi-unit.

    Parameters
    ----------
    fpath: path
        file path
    fname: path
        file name
    shankn: int
        shank number

    Returns
    -------
    df: pd.DataFrame
        has column named 'id' which indicates cluster id and 'time' which
        indicates spike time.

    """
    timing_file = os.path.join(fpath, fname + '.res.' + str(shankn))
    id_file = os.path.join(fpath, fname + '.clu.' + str(shankn))

    timing_df = pd.read_csv(timing_file, names=('time',))
    id_df = pd.read_csv(id_file, names=('id',))
    id_df = id_df[1:]  # the first number is the number of clusters
    noise_inds = ((id_df == 0) | (id_df == 1)).values.ravel()
    df = id_df.join(timing_df)
    df = df.loc[np.logical_not(noise_inds)].reset_index(drop=True)

    df['id'] -= 2

    return df


fname = 'YutaMouse41-150903'
fpath = '/Users/bendichter/Desktop/Buzsaki/SenzaiBuzsaki2017/YutaMouse41-150903'
session_description = 'simulated MEC and LEC data'
identifier = fname
session_start_time = datetime(2015, 7, 31)
institution = 'NYU'
lab = 'Buzsaki'


source=fname
nwbfile = NWBFile(source, session_description, identifier,
                  session_start_time, datetime.now(),
                  institution=institution, lab=lab)
module = nwbfile.create_processing_module(name='0', source=source,
                                          description=source)


channel_groups = get_channel_groups(fpath, fname)
shank_channels = get_shank_channels(fpath, fname)
nshanks = len(shank_channels)
all_shank_channels = np.concatenate(shank_channels)
nchannels = sum(len(x) for x in channel_groups)
lfp_fs = get_lfp_sampling_rate(fpath, fname)

lfp_channel = 0  # value taken from Yuta's spreadsheet

print('reading raw position data...', end='')
pos_df = get_position_data(fpath, fname)
print('done.')

print('setting up raw position data...', end='')
# raw position sensors file
pos0 = nwbfile.add_acquisition(
    SpatialSeries('position sensor0',
                  'raw sensor data from sensor 0',
                  gzip(pos_df[['x0', 'y0']].values),
                  'unknown',
                  timestamps=gzip(pos_df.index.values),
                  resolution=np.nan))

pos1 = nwbfile.add_acquisition(
    SpatialSeries('position sensor1',
                  'raw sensor data from sensor 1',
                  gzip(pos_df[['x1', 'y1']].values),
                  'unknown',
                  timestamps=gzip(pos_df.index.values),
                  resolution=np.nan))
print('done.')

print('setting up electrodes...', end='')
# shank electrodes
electrode_counter = 0
for shankn, channels in zip(range(nshanks), shank_channels):
    device_name = 'shank{}'.format(shankn)
    device = nwbfile.create_device(device_name, fname + '.xml')
    electrode_group = nwbfile.create_electrode_group(
        name=device_name + '_electrodes',
        source=fname + '.xml',
        description=device_name,
        device=device,
        location='unknown')
    for channel in channels:
        nwbfile.add_electrode(channel,
                              np.nan, np.nan, np.nan,  # position?
                              imp=np.nan,
                              location='unknown',
                              filtering='unknown',
                              description='electrode {} of shank {}, channel {}'.format(
                                  electrode_counter, shankn, channel),
                              group=electrode_group)

        if channel == lfp_channel:
            lfp_table_region = nwbfile.create_electrode_table_region(
                [electrode_counter], 'lfp electrode')

        electrode_counter += 1

# special electrodes
device_name = 'special_electrodes'
device = nwbfile.create_device(device_name, fname + '.xml')
electrode_group = nwbfile.create_electrode_group(
    name=device_name + '_electrodes',
    source=fname + '.xml',
    description=device_name,
    device=device,
    location='unknown')
special_electrode_dict = {'ch_wait': 79, 'ch_arm': 78, 'ch_solL': 76,
                          'ch_solR': 77, 'ch_dig1': 65, 'ch_dig2': 68,
                          'ch_entL': 72, 'ch_entR': 71, 'ch_SsolL': 73,
                          'ch_SsolR': 70}
for name, num in special_electrode_dict.items():
    nwbfile.add_electrode(num,
                          np.nan, np.nan, np.nan,
                          imp=np.nan,
                          location='unknown',
                          filtering='unknown',
                          description=name,
                          group=electrode_group)
    nwbfile.create_electrode_table_region([electrode_counter], name)
    electrode_counter += 1

all_table_region = nwbfile.create_electrode_table_region(
    list(range(electrode_counter)), 'all electrodes')
print('done.')

# lfp
print('reading LFPs...', end='')
lfp_file = os.path.join(fpath, fname + '.lfp')
all_channels = np.fromfile(lfp_file, dtype=np.int16).reshape(-1, 80)
all_channels_lfp = all_channels[:, all_shank_channels]
print('done.')

print('making ElectricalSeries objects for LFP...', end='')
all_lfp = nwbfile.add_acquisition(
    ElectricalSeries('all_lfp',
                     'lfp signal for all shank electrodes',
                     gzip(all_channels_lfp),
                     all_table_region,
                     conversion=np.nan,
                     starting_time=0.0,
                     rate=lfp_fs,
                     resolution=np.nan))

lfp = nwbfile.add_acquisition(
    ElectricalSeries('lfp',
                     'signal used as the reference lfp',
                     gzip(all_channels[:, lfp_channel]),
                     lfp_table_region,
                     conversion=np.nan,
                     starting_time=0.0,
                     rate=lfp_fs,
                     resolution=np.nan))
print('done.')

# create epochs corresponding to experiments/environments for the mouse
task_types = ['OpenFieldPosition_ExtraLarge', 'OpenFieldPosition_New_Curtain',
              'OpenFieldPosition_New', 'OpenFieldPosition_Old_Curtain',
              'OpenFieldPosition_Old', 'OpenFieldPosition_Oldlast']

experiment_epochs = []
for label in task_types:
    print('loading normalized position data for ' + label + '...', end='')
    file = os.path.join(fpath, fname + '__' + label)

    matin = loadmat(file)
    tt = matin['twhl_norm'][:, 0]
    pos_data = matin['twhl_norm'][:, 1:3]

    exp_times = find_discontinuities(tt)

    spatial_series_object = SpatialSeries(name=label + ' spatial_series',
                                          source='position sensor0',
                                          data=gzip(pos_data),
                                          reference_frame='unknown',
                                          conversion=np.nan,
                                          resolution=np.nan,
                                          timestamps=gzip(tt))
    pos_obj = Position(source=source, spatial_series=spatial_series_object,
                       name=label + ' position')

    for i, window in enumerate(exp_times):
        experiment_epochs.append(
            nwbfile.create_epoch(source=source,
                                 name=label + '_' + str(i),
                                 start=window[0], stop=window[1]))
    print('done.')

# link epochs to all the relevant timeseries objects

nwbfile.set_epoch_timeseries(experiment_epochs, [pos0, pos1, lfp, all_lfp])


for shank_num in np.arange(1, nshanks + 1):
    print('loading spike times for shank ' + str(shank_num) + '...', end='')
    df = get_clusters_single_shank(fpath, fname, shank_num)
    clu = Clustering(source='source', description='noise and multiunit removed',
                     num=np.array(df['id']), peak_over_rms=[np.nan],
                     times=gzip(np.array(df['time'])),
                     name='shank' + str(shank_num))
    module.add_container(clu)
    print('done.')

print('writing NWB file...', end='')
io = NWBHDF5IO('testy.nwb', mode='w')
io.write(nwbfile)
io.close()
print('done.')
