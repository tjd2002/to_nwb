from pynwb import NWBFile
from datetime import datetime
import numpy as np

nwbfile = NWBFile(source='path of data in old format',  # required
                  session_description='mouse in open exploration and theta maze',  # required
                  identifier='id',  # required
                  session_start_time=datetime(2017, 5, 4, 1, 1, 1),  # required
                  file_create_date=datetime.now(),  # optional
                  experimenter='My Name',  # optional
                  session_id='session_id',  # optional
                  institution='University of My Institution',  # optional
                  lab='My Lab Name',  # optional
                  related_publications='DOI:10.1016/j.neuron.2016.12.011',  # optional
                  )
###

shank_channels = [[0, 1, 2, 3], [0, 1, 2]]

electrode_counter = 0
for shankn, channels in enumerate(shank_channels):
    device_name = 'shank{}'.format(shankn)
    device = nwbfile.create_device(device_name, 'source')
    electrode_group = nwbfile.create_electrode_group(
        name=device_name + '_electrodes',
        source='source',
        description=device_name,
        device=device,
        location='brain area')
    for channel in channels:
        nwbfile.add_electrode(electrode_counter,
                              5.3, 1.5, 8.5,  # position
                              imp=np.nan,
                              location='unknown',
                              filtering='unknown',
                              description='electrode {} of shank {}, channel {}'.format(
                                  electrode_counter, shankn, channel),
                              group=electrode_group)
        electrode_counter += 1
all_table_region = nwbfile.create_electrode_table_region(
    list(range(electrode_counter)), 'all electrodes')


###

from pynwb.ecephys import ElectricalSeries, LFP

lfp_data = np.random.randn(100, 7)

all_lfp = nwbfile.add_acquisition(
    LFP('source',
        ElectricalSeries('name', 'source',
            lfp_data, all_table_region,
            starting_time=0.0, rate=1000.,  # Hz
            resolution=.001,
            conversion=1., unit='V')
        )
    )

###

from pynwb.misc import UnitTimes

# gen spiking data
all_spikes = []
for unit in range(20):
    n_spikes = np.random.poisson(lam=10)
    all_spikes.append(np.random.randn(n_spikes))

# write UnitTimes object
ut = UnitTimes(name='name', source='source')
for i, unit_spikes in enumerate(all_spikes):
    ut.add_spike_times(i, unit_spikes)

spiking_module = nwbfile.create_processing_module(name='spikes',
    source='source', description='data relevant to spiking')

spiking_module.add_container(ut)


###

from pynwb.behavior import SpatialSeries, Position

position_data = np.array([np.linspace(0, 10,100),
                          np.linspace(1, 8, 100)]).T
tt_position = np.linspace(0, 100) / 200

spatial_series_object = SpatialSeries(name='name', source='source',
                                      data=position_data,
                                      reference_frame='unknown',
                                      conversion=np.nan,
                                      resolution=np.nan,
                                      timestamps=tt_position)
pos_obj = Position(source='source', spatial_series=spatial_series_object,
                   name='name')
behavior_module = nwbfile.create_processing_module(name='behavior',
    source='source', description='data relevant to behavior')

behavior_module.add_container(pos_obj)

###

from pynwb.file import Subject

nwbfile.subject = Subject(age='9 months', description='description',
        species='rat', genotype='genotype', sex='M', source='source')

###

from pynwb import NWBHDF5IO

with NWBHDF5IO('test_ephys.nwb', mode='w') as io:
    io.write(nwbfile)

with NWBHDF5IO('test_ephys.nwb', mode='r') as io:
    nwbfile = io.read()

    print(nwbfile.acquisition['LFP']['electrical_series'].data.shape)


