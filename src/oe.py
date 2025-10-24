from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
import pandas as pd


def load_binary(filename, channels, *,
                n_channels=None,
                method=4,
                intype='int16',
                outtype='float32',
                periods=None,
                resample=1,
                return_orig_index=False):
    """
    Load binary file with multi-channel data.
    
    Parameters:
    - filename: path to binary file
    - channels: list of channels to load (starting from 0)
    - n_channels: total channels in file
    - method: loading method (default 4 = memmap)
    - intype: input dtype (e.g., 'int16')
    - outtype: output dtype (e.g., 'float32')
    - periods: optional list of (start, end) tuples (sample indices)
    - resample: int or tuple (p, q) for resampling
    - return_orig_index: return original sample indices
    
    Returns:
    - data: ndarray (n_channels x n_samples)
    - orig_index: optional array of sample indices
    """
    
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"File {filename} not found.")

    dtype_size = np.dtype(intype).itemsize
    file_size = filename.stat().st_size

    if n_channels is None:
        raise ValueError("You must specify n_channels if no .xml or .par file is parsed.")

    n_samples = file_size // (dtype_size * n_channels)

    if isinstance(resample, int):
        resample_pq = (1, resample)
    elif isinstance(resample, (tuple, list)) and len(resample) == 2:
        resample_pq = tuple(resample)
    else:
        raise ValueError("resample must be int or tuple (p, q)")

    data = []
    orig_index = []

    if method == 2:
        # Buffered reading
        buffer_size = 400000
        if resample != 1:
            buffer_size = (buffer_size // resample_pq[1]) * resample_pq[1]

        with open(filename, 'rb') as f:
            if periods is None:
                periods = [(0, n_samples - 1)]

            for start, end in periods:
                f.seek(start * n_channels * dtype_size)
                samples_to_read = end - start + 1
                read = 0
                while read < samples_to_read:
                    count = min(buffer_size, samples_to_read - read)
                    chunk = np.fromfile(f, dtype=intype, count=n_channels * count)
                    chunk = chunk.reshape((-1, n_channels)).T  # shape: (n_channels, count)
                    chunk = chunk[channels, :]
                    if resample != 1:
                        chunk = resample_poly(chunk, resample_pq[0], resample_pq[1], axis=1)
                    data.append(chunk)
                    if return_orig_index:
                        orig = np.arange(start + read, start + read + chunk.shape[1]*resample_pq[1]//resample_pq[0])
                        orig_index.append(orig)
                    read += count

    elif method == 4:
        # Memory-mapped loading
        mmap = np.memmap(filename, dtype=intype, mode='r')
        mmap = mmap.reshape((-1, n_channels)).T  # shape: (n_channels, n_samples)
        mmap = mmap[channels, :]  # select channels

        if periods is None:
            chunk = mmap
            if resample != 1:
                chunk = resample_poly(chunk, resample_pq[0], resample_pq[1], axis=1)
            data = [chunk]
            if return_orig_index:
                orig_index = [np.arange(0, chunk.shape[1])]
        else:
            for start, end in periods:
                chunk = mmap[:, start:end+1]
                if resample != 1:
                    chunk = resample_poly(chunk, resample_pq[0], resample_pq[1], axis=1)
                data.append(chunk)
                if return_orig_index:
                    orig = np.arange(start, end+1, resample_pq[1])
                    orig_index.append(orig)

    else:
        raise ValueError(f"Method {method} not implemented yet")

    data = np.concatenate(data, axis=1).astype(outtype)
    if return_orig_index:
        orig_index = np.concatenate(orig_index)
        return data, orig_index
    return data.T

'''
TTL load

data, idx = load_binary(
    dat_path,
    channels=[384], # which channel/s to load
    n_channels=385,
    method=4,
    intype='int16', #16-bit integers
    outtype='float32',
    resample=1,
    return_orig_index=True
)
'''


def ks_load(ks_path):

    spiketimesfile = ks_path+"/spike_times.npy"  
    spiketimes = np.load(spiketimesfile)

    clusterfile = ks_path+"/spike_clusters.npy"
    spikeclusters = np.load(clusterfile)

    Clusterinfofile = ks_path+"/cluster_info.tsv"
    Clusterinfo = pd.read_csv(Clusterinfofile,sep='\t')

    goodclusts = Clusterinfo['cluster_id'][np.where(Clusterinfo['group']=='good')[0]]

    goods = [] # list of good cluster_id
    for clust in goodclusts :
        goods.append(clust)
    print('Sup! U have...')

    goodspiketimes = {}

    for goodunit in goods : # iterate by good units
        # spike indeces of this good unit
        goodinds = np.where(spikeclusters==goodunit)[0]
        goodspiketimes[goodunit] = spiketimes[goodinds]

    units_ids = list(goodspiketimes.keys())

    print(f'{len(units_ids)} phy-good clusters')

    return goodspiketimes

def acg(spike_times, bin_size_ms=1.0, max_lag_ms=100.0, fs=30000):
    """
    Autocorrelogram for a single spike train (excludes zero-lag self-pairs).
    Spike times are in *samples*.
    """
    s = np.sort(np.asarray(spike_times)) / fs  # seconds
    max_lag_s = max_lag_ms / 1000.0
    bin_edges_ms = np.arange(-max_lag_ms, max_lag_ms + bin_size_ms, bin_size_ms)
    counts = np.zeros(len(bin_edges_ms) - 1, dtype=np.int64)

    for i, t in enumerate(s):
        lo = t - max_lag_s
        hi = t + max_lag_s
        # find neighbors excluding the spike itself (strict inequalities)
        j0 = np.searchsorted(s, lo, side='left')
        j1 = np.searchsorted(s, hi, side='right')
        # exclude index i
        neigh = np.concatenate((s[j0:i], s[i+1:j1]))
        diffs_ms = (neigh - t) * 1000.0
        c, _ = np.histogram(diffs_ms, bins=bin_edges_ms)
        counts += c

    lags_ms = 0.5 * (bin_edges_ms[:-1] + bin_edges_ms[1:])
    return lags_ms, counts



def ccg(spike_times_1, spike_times_2, bin_size_ms=20.0, max_lag_ms=1000.0, fs=30000):
    """
    Cross-correlogram between two spike trains.

    Parameters
    ----------
    spike_times_1, spike_times_2 : array-like
        Spike times in *samples* (ints or floats).
    bin_size_ms : float
        Histogram bin width in milliseconds.
    max_lag_ms : float
        Maximum lag (±) in milliseconds.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    lags_ms : np.ndarray
        Bin centers in milliseconds.
    counts : np.ndarray
        Counts per lag bin.
    """
    # sort & convert to seconds
    s1 = np.sort(np.asarray(spike_times_1)) / fs
    s2 = np.sort(np.asarray(spike_times_2)) / fs

    max_lag_s = max_lag_ms / 1000.0
    bin_edges_ms = np.arange(-max_lag_ms, max_lag_ms + bin_size_ms, bin_size_ms)
    counts = np.zeros(len(bin_edges_ms) - 1, dtype=np.int64)

    # two-sided windowed differences via searchsorted (efficient)
    for t in s1:
        lo = t - max_lag_s
        hi = t + max_lag_s
        i0 = np.searchsorted(s2, lo, side='left')
        i1 = np.searchsorted(s2, hi, side='right')
        diffs_ms = (s2[i0:i1] - t) * 1000.0  # ms
        # histogram into our pre-defined edges
        c, _ = np.histogram(diffs_ms, bins=bin_edges_ms)
        counts += c

    lags_ms = 0.5 * (bin_edges_ms[:-1] + bin_edges_ms[1:])
    return lags_ms, counts