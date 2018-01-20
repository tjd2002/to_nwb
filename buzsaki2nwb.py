import os
from glob import glob

from tqdm import tqdm
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.io import loadmat

from pynwb import NWBFile, NWBHDF5IO
from pynwb.misc import SpikeUnit, UnitTimes
from pynwb.behavior import SpatialSeries, Position

from .utils import find_discontinuities, isin_time_windows


fname = 'YutaMouse41-150903'
fpath = os.path.join('..', 'data', fname)

session_description = 'simulated MEC and LEC data'
identifier = fname
session_start_time = datetime(2015, 7, 31)
institution = 'NYU'
lab = 'Buzsaki'

source = fname


nwbfile = NWBFile(source, session_description, identifier,
                  session_start_time, datetime.now(),
                  institution=institution, lab=lab)


## get experiment names and position data

pos_files = glob(fpath + '/' + fname + '*.mat')

experiments = {}
for file in pos_files:
    if ('StatePeriod' not in file) and ('ROI' not in file):
        lab = file[file.find('__') + 2:-4]
        matin = loadmat(file)
        tt = matin['twhl_norm'][:, 0]
        x = matin['twhl_norm'][:, 1]
        y = matin['twhl_norm'][:, 2]

        exp_times = find_discontinuities(tt)

        data = np.array([x, y]).T
        spatial_series = SpatialSeries('position', source, data, reference_frame='?',
                                       conversion=np.nan, timestamps=tt)
        position = Position(source, spatial_series)

        module = nwbfile.create_processing_module(lab, source, lab)
        module.add_container(position)

        experiments[lab] = {'times': exp_times, 'module': module}

## load celltypes

matin = loadmat(os.path.join('..', 'data', 'DG_all_6__UnitFeatureSummary_add.mat'),
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

## spikes
nshanks = 8

for exp, exp_data in tqdm(experiments.items()):
    spike_units = []
    cell_counter = 0
    for shank_num in np.arange(1, nshanks):
        timing_file = os.path.join(fpath, fname + '.res.' + str(shank_num))
        id_file = os.path.join(fpath, fname + '.clu.' + str(shank_num))

        timing_df = pd.read_csv(timing_file, names=('time',))
        id_df = pd.read_csv(id_file, names=('id',))
        id_df = id_df[1:]  # the first number is the number of clusters
        noise_inds = ((id_df == 0) | (id_df == 1)).values.ravel()
        id_df = id_df.loc[np.logical_not(noise_inds)].reset_index(drop=True)
        timing_df = timing_df.loc[np.logical_not(noise_inds)].reset_index(drop=True)

        for unit_num, df in id_df.join(timing_df).groupby('id'):
            unit_data = df['time'].values / 20000
            unit_data = unit_data[isin_time_windows(unit_data, exp_data['times'])]
            spike_units.append(SpikeUnit(name=str(unit_ids[cell_counter]),
                                         times=unit_data,
                                         unit_description=celltype_names[cell_counter],
                                         source='S{}C{}'.format(shank_num, unit_num)))
            cell_counter += 1
        unit_times = UnitTimes(source, spike_units, name='spikes')
        exp_data['module'].add_container(unit_times)

# write NWB file
io = NWBHDF5IO(os.path.join('..', 'data', fname + '.nwb'),  mode='w')
io.write(nwbfile)
io.close()
