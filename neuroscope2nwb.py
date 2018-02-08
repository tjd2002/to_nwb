from bs4 import BeautifulSoup
import os
import numpy as np
import pandas as pd
from glob import glob
from scipy.io import loadmat

from datetime import datetime
from pynwb import NWBFile, NWBHDF5IO
from pynwb.ephys import ElectricalSeries, LFP, Clustering
form pynwb import SpatialSeries

from .utils import find_discontinuities


def load_xml(filepath):
    with open(filepath, 'r') as xml_file:
        contents = xml_file.read()
        soup = BeautifulSoup(contents, 'xml')
    return soup


def get_channel_groups(fpath, fname):
    xml_filepath = os.path.join(fpath, fname + '.xml')
    soup = load_xml(xml_filepath)

    channel_groups = [[int(channel.string)
                       for channel in group.find_all('channel')]
                      for group in soup.channelGroups.find_all('group')]

    return channel_groups


def get_shank_channels(fpath, fname):
    xml_filepath = os.path.join(fpath, fname + '.xml')
    soup = load_xml(xml_filepath)

    shank_channels = [[int(channel.string)
                       for channel in group.find_all('channel')]
                      for group in soup.spikeDetection.channelGroups.find_all('group')]
    return shank_channels


def get_lfp_sampling_rate(fpath, fname):
    xml_filepath = os.path.join(fpath, fname + '.xml')
    soup = load_xml(xml_filepath)

    return float(soup.lfpSamplingRate.string)


def get_position_data(fpath, fname, fs=1250./32.):
    df = pd.read_csv(os.path.join(fpath, fname + '.whl'),
                     sep='\t', names=('x0', 'y0', 'x1', 'y1'))

    df.index = np.arange(len(df)) / fs
    df.index.name = 'tt (sec)'

    return df


def get_clusters(fpath, fname, shankn):
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
fpath = '../../Buzsaki/SenzaiBuzsaki2017/' + fname
session_description = 'simulated MEC and LEC data'
identifier = fname
session_start_time = datetime(2015, 7, 31)
institution = 'NYU'
lab = 'Buzsaki'


source=fname
nwbfile = NWBFile(source, session_description, identifier,
                  session_start_time, datetime.now(),
                  institution=institution, lab=lab)
module = nwbfile.create_processing_module(name='0', source=source, description=source)


channel_groups = get_channel_groups(fpath, fname)
shank_channels = get_shank_channels(fpath, fname)
nshanks = len(shank_channels)
all_shank_channels = np.concatenate(shank_channels)
nchannels = sum(len(x) for x in channel_groups)
lfp_fs = get_lfp_sampling_rate(fpath, fname)

lfp_channel = 0  # value taken from Yuta's spreadsheet

pos_df = get_position_data(fpath, fname)

# raw position sensors file
pos0 = nwbfile.add_acquisition(
    SpatialSeries('sensor0',
                  'raw sensor data from sensor 0',
                  pos_df[['x0', 'y0']].values,
                  'unknown',
                  timestamps=pos_df.index.values,
                  resolution=np.nan))


pos1 = nwbfile.add_acquisition(
    SpatialSeries('sensor1',
                  'raw sensor data from sensor 1',
                  pos_df[['x1', 'y1']].values,
                  'unknown',
                  timestamps=pos_df.index.values,
                  resolution=np.nan))


# electrodes
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
                              location='blank',
                              filtering='blank',
                              description='electrode {} of shank {}, channel {}'.format(
                                  electrode_counter, shankn, channel),
                              group=electrode_group)

        if channel == lfp_channel:
            lfp_table_region = nwbfile.create_electrode_table_region(
                [electrode_counter], 'lfp electrode')

        electrode_counter += 1

all_table_region = nwbfile.create_electrode_table_region(
    list(range(electrode_counter)), 'all electrodes')



# lfp
lfp_file = os.path.join(fpath, fname + '.lfp')

all_channels = np.fromfile(lfp_file, dtype=np.int16).reshape(-1, 80)
all_channels_lfp = all_channels[:, all_shank_channels]

all_lfp = nwbfile.add_acquisition(
    ElectricalSeries('all_lfp',
                     'lfp signal for all shank electrodes',
                     all_channels_lfp,
                     all_table_region,
                     conversion=np.nan,
                     starting_time=0.0,
                     rate=lfp_fs,
                     resolution=np.nan))

lfp = nwbfile.add_acquisition(
    ElectricalSeries('lfp',
                     'signal unsed as the reference lfp',
                     all_channels[:, lfp_channel],
                     lfp_table_region,
                     conversion=np.nan,
                     starting_time=0.0,
                     rate=lfp_fs,
                     resolution=np.nan))

# create epochs corresponding to experiments/environments for the mouse
pos_files = glob(fpath + '/' + fname + '*.mat')

experiments = {}
for file in pos_files:
    if ('StatePeriod' not in file) and ('ROI' not in file) and \
            ('lfpphase' not in file) and ('session' not in file):
        lab = file[file.find('__') + 2:-4]
        matin = loadmat(file)
        tt = matin['twhl_norm'][:, 0]

        exp_times = find_discontinuities(tt)

        experiment_epochs = []
        for i, window in enumerate(exp_times):
            experiment_epochs.append(
                nwbfile.create_epoch(source=source,
                                     name=lab + '_' + str(i),
                                     start=window[0], stop=window[1]))

# link epochs to all the relevant timeseries objects
for ts in (pos0, pos1, all_channels, lfp):
    nwbfile.set_epoch_timeseries(experiment_epochs, ts)


for shank_num in np.arange(1, nshanks):
    df = get_clusters(fpath, fname, shank_num)
    clu = Clustering(source='source', description='noise and multiunit removed',
                     num=df['id'], peak_over_rms=[np.nan], times=df['time'],
                     name='shank' + str(shank_num))
    module.add_container(clu)


io = NWBHDF5IO(fname + '.nwb', mode='w')
io.write(nwbfile)
io.close()
