from pynwb.spec import NWBDatasetSpec, NWBNamespaceBuilder, NWBGroupSpec


def main():
    name = 'buzsaki'
    ns_path = name + '.namespace.yaml'
    ext_source = name + '.extensions.yaml'

    gid_spec = NWBDatasetSpec(doc='global id for neuron',
                              shape=(None, 1),
                              name='cell_index', dtype='int')
    data_val_spec = NWBDatasetSpec(doc='Data values indexed by pointer',
                                   shape=(None, 1),
                                   name='value', dtype='float')
    data_pointer_spec = NWBDatasetSpec(doc='Pointers that index data values',
                                       shape=(None, 1),
                                       name='pointer', dtype='int')

    gid_pointer_value_spec = [gid_spec, data_val_spec, data_pointer_spec]

    cat_cell_info = NWBGroupSpec(neurodata_type_def='CatCellInfo',
                                 doc='Categorical Cell Info',
                                 datasets=[gid_spec,
                                           NWBDatasetSpec(name='indices',
                                                          doc='indices into values for each gid in order',
                                                          shape=(None, 1),
                                                          dtype='int'),
                                           NWBDatasetSpec(name='values',
                                                          doc='list of unique values',
                                                          shope=(None, 1), dtype='str')],
                                 neurodata_type_inc='NWBDataInterface')

    population_spike_times = NWBGroupSpec(neurodata_type_def='PopulationSpikeTimes',
                                          doc='Population Spike Times',
                                          datasets=gid_pointer_value_spec,
                                          neurodata_type_inc='NWBDataInterface')

    ns_builder = NWBNamespaceBuilder(name + ' extensions', name)
    for spec in [population_spike_times, cat_cell_info]:
        ns_builder.add_spec(ext_source, spec)
    ns_builder.export(ns_path)


if __name__ == "__main__":
    main()
