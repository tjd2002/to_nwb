from scipy.io import loadmat
from datetime import datetime

from pynwb.ecephys import ElectricalSeries, Clustering
from pynwb import NWBFile, NWBHDF5IO

from general import gzip

fpath = '/Users/bendichter/Desktop/Schnitzer/data/eoPHYS_SS1anesthesia/converted_data.mat'
fname = 'ex_simon'
session_description = ''
identifier = fname
institution = 'Stanford'
lab = 'Schnitzer'
source = fname

matin = loadmat(fpath, struct_as_record=False)
data = matin['data'][0]
session_start_time = data[0].abstime

nwbfile = NWBFile(source, session_description, identifier,
                  session_start_time, datetime.now(),
                  institution=institution, lab=lab)

module = nwbfile.create_processing_module(name='0', source=source,
                                          description=source)

for trial_data in data:

    trial_data.ephys
    trial_data.time
    trial_data.abstime
    trial_data.events
    trial_data.tempo_data



