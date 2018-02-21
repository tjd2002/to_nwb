from functools import partialmethod

from pynwb import load_namespaces
from pynwb import get_class

import numpy as np

from pynwb.form.backends.hdf5 import H5DataIO as gzip
gzip.__init__ = partialmethod(gzip.__init__, compress=True)

# load custom classes
name = 'general'
ns_path = name + '.namespace.yaml'
ext_source = name + '.extensions.yaml'
load_namespaces(ns_path)
CatCellInfo = get_class('CatCellInfo', name)
PopulationSpikeTimes = get_class('PopulationSpikeTimes', name)


def build_cat_cell_info(data, source, name='cell type'):
    """

    Parameters
    ----------
    data: iterable
    source:
    name:

    Returns
    -------
    Cat_Cell_Info object

    """

    u_cats, indices = np.unique(data, return_inverse=True)
    cci_obj = CatCellInfo(name, source, values=u_cats, indices=indices,
                          cell_index=list(range(len(indices))))

    return cci_obj






