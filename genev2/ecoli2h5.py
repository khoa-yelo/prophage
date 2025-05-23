#!/usr/bin/env python3
"""
Convert a dict of per‐sample numpy arrays into an HDF5 file
with datasets for mean, max, mean_middle, max_middle,
and a mapping from row‐index to sample name.
"""

import h5py
import numpy as np

def dict_to_h5(data_dict, out_file):
    # preserve insertion order of samples
    sample_names = list(data_dict.keys())
    n_samples = len(sample_names)

    # collect arrays
    means = []
    maxs = []
    mean_middles = []
    max_middles = []

    # keys in your dict
    MEAN_KEY = 'mean'
    MAX_KEY = 'max'
    MEAN_MIDDLE_KEY = 'mean_middle_layer_12'
    MAX_MIDDLE_KEY = 'max_middle_layer_12'

    for name in sample_names:
        item = data_dict[name]
        means.append(item[MEAN_KEY])
        maxs.append(item[MAX_KEY])
        mean_middles.append(item[MEAN_MIDDLE_KEY])
        max_middles.append(item[MAX_MIDDLE_KEY])

    # stack into 2D arrays: (n_samples, L)
    means = np.stack(means, axis=0)
    maxs = np.stack(maxs, axis=0)
    mean_middles = np.stack(mean_middles, axis=0)
    max_middles = np.stack(max_middles, axis=0)

    # write to HDF5
    with h5py.File(out_file, 'w') as f:
        f.create_dataset('mean', data=means)
        f.create_dataset('max', data=maxs)
        f.create_dataset('mean_middle', data=mean_middles)
        f.create_dataset('max_middle', data=max_middles)

        # store sample‐name lookup
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('samples',
                         data=np.array(sample_names, dtype=dt),
                         dtype=dt)

    print(f"Wrote {n_samples} samples to '{out_file}'")
    
with open(ecoli_embeddings, "rb") as f:
    ecoli_data = pickle.load(f)
dict_to_h5(ecoli_data, "ecoli_embeddings.h5")