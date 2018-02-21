import os
from functools import partialmethod

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from pynwb.form.backends.hdf5 import H5DataIO as gzip
gzip.__init__ = partialmethod(gzip.__init__, compress=True)

from pynwb import load_namespaces, get_class

name = 'general'
ns_path = name + '.namespace.yaml'

load_namespaces(ns_path)
PopulationSpikeTimes = get_class('PopulationSpikeTimes', name)


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


def build_pop_spikes(fpath, fname, shanks=None, name='PopulationSpikeTimes',
                     source=None, compress=True):
    """Convert from .res and .clu files to parameters that go into
    PopulationSpikeTimes

    Parameters
    ----------
    fpath: str
    fname: str
    shanks: None | list(ints)
        shank numbers to process. If None, use 1:8
    name: str
    source: str

    Returns
    -------
    PopulationSpikeTimes Object

    """
    fnamepath = os.path.join(fpath, fname)

    if shanks is None:
        shanks = range(1, 9)

    if source is None:
        source = fnamepath + '.res.*; ' + fnamepath + '.clu.*'

    values = []
    pointers = [0, ]
    for shank_num in shanks:
        df = get_clusters_single_shank(fpath, fname, shank_num)
        for cluster_num, idf, in df.groupby('id'):
            values += list(idf['time'])
            pointers.append(len(values))
    cell_index = np.arange(len(pointers) - 1)

    if compress:
        cell_index = gzip(cell_index)
        values = gzip(values)
        pointers = gzip(pointers)

    pst_obj = PopulationSpikeTimes(cell_index=cell_index, value=values,
                                   pointer=pointers, name=name, source=source)

    return pst_obj
