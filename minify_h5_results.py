"""
This script extracts the longest time series from all HDF5 files matching the pattern *_j*.h5
found in a given directory, and combines the extracted data into a new HDF5 file with the same
name as the deepest directory in the path hierarchy. The output file is stored at the top level
of the specified directory.
"""

import os
import sys

import h5py
import numpy as np


def extract_longest_timeseries(path_name: str):
    """
    Retrieve the longest time series from all HDF5 files matching the pattern *_j*.h5
    found in the directory specified by `path_name`. Combine the extracted data into a
    new HDF5 file with the same name as the deepest directory in the path hierarchy and
    store it at the top level of the specified directory.

    Args:
        path_name (str): The path of the directory containing the HDF5 files.

    Returns:
        None.
    """
    f_min_filepath = os.path.basename(os.path.normpath(path_name))
    f_min = h5py.File(os.path.normpath(path_name) + '/' + f_min_filepath + '.h5', 'w')

    for subdir, _, files in os.walk(path_name):
        for file in files:
            filepath = os.path.join(subdir, file)
            if '_j' not in file:
                continue
            print(filepath)
            with h5py.File(filepath, 'r') as f_part:
                for key, value in f_part.items():
                    # TODO: Store file number as attribute in combined file
                    t_len = len(value.attrs['times'])
                    if t_len == np.array(value).shape[2]:
                        f_part.copy(f"/{key}", f_min["/"])
    f_min.close()


if __name__ == '__main__':
    PATH_NAME = sys.argv[1]
    extract_longest_timeseries(PATH_NAME)
