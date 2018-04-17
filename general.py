from functools import partialmethod

from pynwb import load_namespaces
from pynwb import get_class

from pynwb.form.backends.hdf5 import H5DataIO as gzip
gzip.__init__ = partialmethod(gzip.__init__, compress=True)

# load custom classes
name = 'general'
ns_path = name + '.namespace.yaml'
ext_source = name + '.extensions.yaml'
load_namespaces(ns_path)

CatCellInfo = get_class('CatCellInfo', name)


