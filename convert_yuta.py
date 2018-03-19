import os

from datetime import datetime
import numpy as np
from scipy.io import loadmat

from pynwb import NWBFile, NWBHDF5IO
from pynwb.behavior import SpatialSeries, Position
from pynwb.ecephys import ElectricalSeries, Clustering

from utils import find_discontinuities
from neuroscope import (get_channel_groups, get_shank_channels,
                        get_lfp_sampling_rate, get_position_data, gzip,
                        get_clusters_single_shank, build_unit_times)

from general import CatCellInfo

WRITE_ALL_LFPS = False


fpath = '/Users/bendichter/Desktop/Buzsaki/SenzaiBuzsaki2017/YutaMouse41-150903'
fpath_base, fname = os.path.split(fpath)
session_description = 'mouse in open exploration and circular 8-maze'
identifier = fname
session_start_time = datetime(2015, 7, 31)
institution = 'NYU'
lab = 'Buzsaki'


source = fname
nwbfile = NWBFile(source, session_description, identifier,
                  session_start_time, datetime.now(),
                  institution=institution, lab=lab)
module_behavior = nwbfile.create_processing_module(name='behavior',
                                                   source=source,
                                                   description=source)

all_ts = []

channel_groups = get_channel_groups(fpath, fname)
shank_channels = get_shank_channels(fpath, fname)
nshanks = len(shank_channels)
all_shank_channels = np.concatenate(shank_channels)
nchannels = sum(len(x) for x in channel_groups)
lfp_fs = get_lfp_sampling_rate(fpath, fname)

lfp_channel = 0  # value taken from Yuta's spreadsheet

print('reading raw position data...', end='', flush=True)
pos_df = get_position_data(fpath, fname)
print('done.')

print('setting up raw position data...', end='', flush=True)
# raw position sensors file
pos0 = nwbfile.add_acquisition(
    SpatialSeries('position sensor0',
                  'raw sensor data from sensor 0',
                  gzip(pos_df[['x0', 'y0']].values),
                  'unknown',
                  timestamps=gzip(pos_df.index.values),
                  resolution=np.nan))
all_ts.append(pos0)

pos1 = nwbfile.add_acquisition(
    SpatialSeries('position sensor1',
                  'raw sensor data from sensor 1',
                  gzip(pos_df[['x1', 'y1']].values),
                  'unknown',
                  timestamps=gzip(pos_df.index.values),
                  resolution=np.nan))
all_ts.append(pos1)
print('done.')

print('setting up electrodes...', end='', flush=True)
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
print('reading LFPs...', end='', flush=True)
lfp_file = os.path.join(fpath, fname + '.lfp')
all_channels = np.fromfile(lfp_file, dtype=np.int16).reshape(-1, 80)
all_channels_lfp = all_channels[:, all_shank_channels]
print('done.')

if WRITE_ALL_LFPS:
    print('making ElectricalSeries objects for LFP...', end='', flush=True)
    all_lfp = nwbfile.add_acquisition(
        ElectricalSeries('all_lfp',
                         'lfp signal for all shank electrodes',
                         gzip(all_channels_lfp),
                         all_table_region,
                         conversion=np.nan,
                         starting_time=0.0,
                         rate=lfp_fs,
                         resolution=np.nan))
    all_ts.append(all_lfp)


lfp = nwbfile.add_acquisition(
    ElectricalSeries('lfp',
                     'signal used as the reference lfp',
                     gzip(all_channels[:, lfp_channel]),
                     lfp_table_region,
                     conversion=np.nan,
                     starting_time=0.0,
                     rate=lfp_fs,
                     resolution=np.nan))
all_ts.append(lfp)
print('done.')

# create epochs corresponding to experiments/environments for the mouse
task_types = ['OpenFieldPosition_ExtraLarge', 'OpenFieldPosition_New_Curtain',
              'OpenFieldPosition_New', 'OpenFieldPosition_Old_Curtain',
              'OpenFieldPosition_Old', 'OpenFieldPosition_Oldlast']

experiment_epochs = []
for label in task_types:
    print('loading normalized position data for ' + label + '...', end='', flush=True)
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

    module_behavior.add_container(pos_obj)

    for i, window in enumerate(exp_times):
        experiment_epochs.append(
            nwbfile.create_epoch(source=source,
                                 name=label + '_' + str(i),
                                 start=window[0], stop=window[1]))
    print('done.')

# link epochs to all the relevant timeseries objects

nwbfile.set_epoch_timeseries(experiment_epochs, all_ts)

module_clustering = nwbfile.create_processing_module(name='clustering',
                                                     source=source,
                                                     description=source)
for shank_num in np.arange(1, nshanks + 1):
    print('loading spike times for shank ' + str(shank_num) + '...', end='')
    df = get_clusters_single_shank(fpath, fname, shank_num)
    clu = Clustering(source='source', description='noise and multiunit removed',
                     num=np.array(df['id']), peak_over_rms=[np.nan],
                     times=gzip(np.array(df['time'])),
                     name='shank' + str(shank_num))
    module_clustering.add_container(clu)
    print('done.')

## load celltypes
matin = loadmat(os.path.join(fpath_base, 'DG_all_6__UnitFeatureSummary_add.mat'),
                struct_as_record=False)['UnitFeatureCell'][0][0]

# taken from ReadMe
celltype_dict = {
    0: 'unknown',
    1: 'granule cells (DG) or pyramidal cells (CA3)  (need to use region info. see below.)',
    2: 'mossy cell',
    3: 'narrow waveform cell',
    4: 'optogenetically tagged SST cell',
    5: 'wide waveform cell (narrower, exclude opto tagged SST cell)',
    6: 'wide waveform cell (wider)',
    8: 'positive waveform unit (non-bursty)',
    9: 'positive waveform unit (bursty)',
    10: 'positive negative waveform unit'
}

region_dict = {3: 'CA3', 4: 'DG'}

this_file = matin.fname == fname
celltype_ids = matin.fineCellType.ravel()[this_file]
region_ids = matin.region.ravel()[this_file]
unit_ids = matin.unitID.ravel()[this_file]

celltype_names = []
for celltype_id, region_id in zip(celltype_ids, region_ids):
    if celltype_id == 1:
        if region_id == 3:
            celltype_names.append('pyramidal cell')
        elif region_id == 4:
            celltype_names.append('granule cell')
        else:
            raise Exception('unknown type')
    else:
        celltype_names.append(celltype_dict[celltype_id])

u_cats, indices = np.unique(celltype_names, return_inverse=True)

cci_obj = CatCellInfo(name='Cell Types',
                      source='DG_all_6__UnitFeatureSummary_add.mat',
                      values=u_cats, indices=indices,
                      cell_index=list(range(len(indices))))

ut_obj = build_unit_times(fpath, fname)

module_spikes = nwbfile.create_processing_module('spikes', source=source,
                                                 description=source)

module_spikes.add_container(ut_obj)
module_spikes.add_container(cci_obj)

out_fname = 'yuta_data.nwb'
print('writing NWB file...', end='', flush=True)
with NWBHDF5IO(out_fname, mode='w') as io:
    io.write(nwbfile, cache_spec=False)
print('done.')

print('testing read...', end='', flush=True)
# test read
with NWBHDF5IO(out_fname, mode='r') as io:
    io.read()
print('done.')

