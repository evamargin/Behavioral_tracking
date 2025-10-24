# here I m collecting patterns I repeat again and again --> time to wrap it up

import numpy as np
from pathlib import Path
import os
import pickle
import sys
sys.path.append('/storage3/eva/code/remapping/src')

def path_load(date, animal,p=True):
    """
    returns dat_path, ks_path, csv_path, res_path
    """

    dat_path = f'/storage3/eva/data/raw/oe/{animal}/{animal}_{date}/continuous.dat'
    ks_path = f'/storage3/eva/data/processed/{animal}/{animal}_{date}/kilosort'
    csv_path = f'/storage3/eva/data/processed/{animal}/{animal}_{date}/motive/out_csv'
    res_path = f'/storage3/eva/code/neuropixels/results/{animal}/{animal}_{date}'

    if p==True:
        print('Ola! For folders inside of res_path: Path(res_path)/"folder_name"')
        print('To make new folder (if not exist): your_path.mkdir(parents=True, exist_ok=True)')

    return dat_path, ks_path, csv_path, res_path      


def accumarray(spike_times_dict, sampling_rate=30000, bin_size_ms=300, step_size_ms=100):
    '''
    default params for umap: sliding 300ms window with 100ms step

    usage:
        spike_matrix = {}
        bin_centers_sec = {}
        cell_ids = {}

        for of in of_keys:
            spike_matrix[of], bin_centers_sec[of], cell_ids[of] = accumarray(spikes_by_periods[of])

        accumarray4umap = {}
        accumarray4umap['spike_matrix'] = spike_matrix
        accumarray4umap['bin_centers_sec'] = bin_centers_sec
        accumarray4umap['cell_ids'] = cell_ids

        with open(f"{pa_path}/accumarray4umap_{date}.pkl", "wb") as f:
            pickle.dump(accumarray4umap, f)    
    '''
    bin_size = int(bin_size_ms / 1000 * sampling_rate)
    step_size = int(step_size_ms / 1000 * sampling_rate)

    # Get all spike times to determine bin range
    max_time = max([np.max(spikes) if len(spikes) > 0 else 0 for spikes in spike_times_dict.values()])
    bin_starts = np.arange(0, max_time - bin_size + 1, step_size)
    n_bins = len(bin_starts)
    cell_ids = sorted(spike_times_dict.keys())
    n_cells = len(cell_ids)

    # Initialize output matrix: (n_cells, n_bins)
    spike_matrix = np.zeros((n_cells, n_bins), dtype=int)

    for i, cell_id in enumerate(cell_ids):
        spikes = np.array(spike_times_dict[cell_id])
        for j, start in enumerate(bin_starts):
            end = start + bin_size
            spike_matrix[i, j] = np.sum((spikes >= start) & (spikes < end))

    # Bin centers in seconds
    bin_centers_sec = (bin_starts + bin_size // 2) / sampling_rate

    return spike_matrix, bin_centers_sec, cell_ids

def neuronal_classes(date, goodspiketimes, res_path):
    # neuronal classes
    try:
        with open(Path(res_path) / f"unit_classes_{date}.pkl", "rb") as f:
            unit_class_dict = pickle.load(f)
        
        all = list(goodspiketimes.keys())
        pyr = [u for u in all if unit_class_dict[u] == "pyr"]
        inter = [u for u in all if unit_class_dict[u] == "int"]
    except:
        print('units were not classified')

    return all, pyr, inter