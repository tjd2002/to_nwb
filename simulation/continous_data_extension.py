import numpy as np
from datetime import datetime
import h5py

from pynwb.spec import NWBGroupSpec, NWBDatasetSpec, NWBAttributeSpec, NWBNamespaceBuilder
from pynwb import get_class, load_namespaces, NWBHDF5IO, NWBFile


project_name = 'simulationoutput'
ns_path = project_name + '.namespace.yaml'
ext_source = project_name + '.extensions.yaml'
ns_builder = NWBNamespaceBuilder('SimulationOutput', project_name)


def build_ext():
    datasets = [
        NWBDatasetSpec(doc='Contiguous multi-cell, multi-compartmental data',
                       # dtype=['array_data', 'float'],
                       dtype='float', #float64',
                       shape=(None, None),
                       dim=('time_steps', 'n_components'),
                       name='data'),
        NWBDatasetSpec(doc='list of cell ids',
                       dtype='uint32',
                       shape=(None, 1),
                       name='gid',
                       quantity='?'),
        NWBDatasetSpec(doc='index pointer',
                       dtype='uint64',
                       shape=(None, 1),
                       name='index_pointer'),
        NWBDatasetSpec(doc='cell compartment ids corresponding to a given column in the data',
                       dtype='uint32',
                       shape=(None, 1),
                       name='element_id'),
        NWBDatasetSpec(doc='relative position of recording within a given compartment',
                       dtype='float',
                       shape=(None, 1),
                       name='element_pos')
    ]

    attributes = [
        NWBAttributeSpec('cell_var', 'variable being recorded in the data table', 'text')
    ]

    cont_data = NWBGroupSpec(doc='A spec for storing cell recording variables',
                             attributes=attributes,
                             datasets=datasets,
                             neurodata_type_inc='NWBDataInterface',
                             neurodata_type_def='VarTable')

    ns_builder.add_spec(ext_source, cont_data)
    ns_builder.export(ns_path)


def build_toy_example():
    load_namespaces(ns_path)
    VarTable = get_class('VarTable', project_name)
    vmtable = VarTable(source='source',
                       name='vm_table',
                       gid=[1, 2, 3],
                       index_pointer=[1, 2, 3],
                       cell_var='Membrane potential (mV)',
                       # data_table=np.zeros((10, 10), dtype=np.float),
                       data=[[1., 2., 4.], [1., 2., 4.]],
                       element_id=[0, 0, 0],
                       element_pos=[1., 1., 1.])

    nwbfile = NWBFile(source='source', session_description='session_description', identifier='identifier',
                      session_start_time=datetime.now(), file_create_date=datetime.now(), institution='institution',
                      lab='lab')

    module = nwbfile.create_processing_module(name='name', source='source', description='description')
    module.add_container(vmtable)

    # with NWBHDF5IO('membrane_potential.nwb', mode='w') as io:
    #    io.write(nwbfile)
    io = NWBHDF5IO('mem_potential_toy.nwb', mode='w')
    io.write(nwbfile)
    io.close()


def build_real_example():
    load_namespaces(ns_path)
    VarTable = get_class('VarTable', project_name)
    input_data = h5py.File('sim_data/cell_vars.h5', 'r')
    vmtable = VarTable(source='source',
                       name='vm_table',
                       data=np.array(input_data['/v/data']),
                       gid=np.array(input_data['mapping/gids']), #[1, 2, 3],
                       index_pointer=np.array(input_data['mapping/index_pointer']), #'[1, 2, 3],
                       cell_var='Membrane potential (mV)',
                       element_id=np.array(input_data['mapping/element_id']), #[0, 0, 0],
                       element_pos=np.array(input_data['mapping/element_pos']))  #[1., 1., 1.])

    nwbfile = NWBFile(source='source', session_description='session_description', identifier='identifier',
                      session_start_time=datetime.now(), file_create_date=datetime.now(), institution='institution',
                      lab='lab')

    module = nwbfile.create_processing_module(name='name', source='source', description='description')
    module.add_container(vmtable)

    io = NWBHDF5IO('mem_potential_toy.nwb', mode='w')
    io.write(nwbfile)
    io.close()

if __name__ == '__main__':
    build_ext()
    # build_toy_example()
    build_real_example()
