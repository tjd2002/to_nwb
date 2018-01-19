import os

from h5py import File
from tqdm import tqdm
from datetime import datetime
import numpy as np

from pynwb import NWBFile, NWBHDF5IO
from pynwb.misc import SpikeUnit, UnitTimes
from pynwb.behavior import SpatialSeries, Position

from .utils import pairwise


def convert_file(fpath, session_start_time,
                 session_description='simulated MEC and LEC data'):

    fname = os.path.split(fpath)[1]
    identifier = fname[:-4]
    institution = 'Stanford'
    lab = 'Soltesz'
    source=fname[:-4]

    # extract data
    spike_units = []
    with File(fpath, 'r') as f:
        for cell_type in ('MPP', 'LPP'):
            spiketrain = f['Populations'][cell_type]['Vector Stimulus 0']['spiketrain']
            for i, (start, fin) in tqdm(enumerate(pairwise(spiketrain['Attribute Pointer'])),
                                        total=len(spiketrain['Attribute Pointer']),
                                        desc=cell_type):
                if not (start == fin):
                    UnitData = spiketrain['Attribute Value'][start:fin] / 1000
                    spike_units.append(SpikeUnit(name=cell_type + '{:05d}'.format(i),
                                                 times=UnitData,
                                                 unit_description=cell_type,
                                                 source=source))

        ## Position
        x = f['Trajectory 0']['x']
        y = f['Trajectory 0']['y']
        rate = 1 / (f['Trajectory 0']['t'][1] - f['Trajectory 0']['t'][0]) * 1000

        pos_data = np.array([x, y]).T


    # write to NWB
    nwbfile = NWBFile(source, session_description, identifier,
                      session_start_time, datetime.now(),
                      institution=institution, lab=lab)

    rf_module = nwbfile.create_processing_module('receptive fields', source, 'spike times')

    spatial_series = SpatialSeries('Position',
                                   source, pos_data,
                                   reference_frame='NA',
                                   conversion=1 / 100.,
                                   resolution=0.1,
                                   starting_time=0.0,
                                   rate=rate)

    behav_ts = Position(source, spatial_series)
    unit_times = UnitTimes(source, spike_units, name='simulated cell spike data')

    rf_module.add_container(unit_times)
    rf_module.add_container(behav_ts)


