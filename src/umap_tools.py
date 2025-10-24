# here is my playground functions for umap cz i m tired of writing them again and again
# depends on oe.py and motive.py, utils.py
# umap_utils.py

import numpy as np
from pathlib import Path
import os
import pickle
import sys
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Add project src to path if needed
sys.path.append('/storage3/eva/code/remapping/src')

import oe
import motive
import utils

# Sampling rate for position data
FS_POS = 120

# ------------------------------
# I/O and helper functions
# ------------------------------

def load_spike_times(ks_path):
    return oe.ks_load(ks_path)

def load_behav_periods(animal, date, res_path):
    path = Path(res_path) / f"preprocessing/behav_periods_{animal}_{date}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)

def load_meta(date, res_path):
    with open(Path(res_path) / "preprocessing" / f"meta_{date}.pkl", "rb") as f:
        return pickle.load(f)

def load_smooth_pitch_dict(date, res_path):
    with open(Path(res_path) / "preprocessing" / f"smooth_pitch_dict_{date}.pkl", "rb") as f:
        return pickle.load(f)

def prepare_umap_inputs(animal, date, ks_path, res_path, csv_path):
    """
    Prepares data for UMAP analysis.

    Returns dictionary with spike matrix, position, pitch, speed:
    spike_matrix	all bins, unfiltered
    pitch	full resampled pitch
    pos_data	full resampled 3D position
    speeds	full speed traces
    *_run	same variables but filtered (movement/rearing only)
    """

    # Load data
    goodspiketimes = oe.ks_load(ks_path)
    with open(Path(res_path) / f"preprocessing/behav_periods_{animal}_{date}.pkl", "rb") as f:
        periods = pickle.load(f)
    with open(Path(res_path) / "preprocessing" / f"meta_{date}.pkl", "rb") as f:
        meta = pickle.load(f)
    csv_dict = motive.get_csv_dict(csv_path)

    all_keys = list(periods.keys())
    of_keys = [k for k in periods if "of" in k]
    sl_keys = [k for k in periods if "sl" in k]

    # Prepare spike data split by periods
    def extract_spikes_by_periods(keys):
        spikes_by_periods = {}
        for key in keys:
            start, end = periods[key]
            spikes_by_periods[key] = {
                unit: goodspiketimes[unit][(goodspiketimes[unit] >= start) & (goodspiketimes[unit] <= end)] - start
                for unit in goodspiketimes
            }
        return spikes_by_periods

    spikes_by_periods = extract_spikes_by_periods(all_keys)

    # Load or compute accumarray
    accum_path = Path(res_path) / f"pa/accumarray4umap_extended_{date}.pkl"
    if not accum_path.exists():
        print("Running accumarray... be patient :)")
        spike_matrix, bin_centers_sec, cell_ids = {}, {}, {}
        for key in all_keys:
            spike_matrix[key], bin_centers_sec[key], cell_ids[key] = utils.accumarray(spikes_by_periods[key])
        accumarray4umap = {
            'spike_matrix': spike_matrix,
            'bin_centers_sec': bin_centers_sec,
            'cell_ids': cell_ids
        }
        with open(accum_path, "wb") as f:
            pickle.dump(accumarray4umap, f)
    else:
        with open(accum_path, "rb") as f:
            accumarray4umap = pickle.load(f)

    spike_matrix = accumarray4umap['spike_matrix']
    bin_centers_sec = accumarray4umap['bin_centers_sec']
    cell_ids = accumarray4umap['cell_ids']

    # Load positional and pitch data
    with open(Path(res_path) / "preprocessing" / f"smooth_pitch_dict_{date}.pkl", "rb") as f:
        smooth_pitch_dict = pickle.load(f)

    pos_data_interp, frame_times_all = {}, {}
    for key in meta:
        df = csv_dict[meta[key]]
        frame_times_all[key] = motive.get_frame_times(df)
        _, arrays_interp = motive.get_arrays(df, metric='Position', dim_array=['X', 'Y', 'Z'], interpolate=True)
        pos_data_interp[key] = arrays_interp

    # Resample pitch and position to neural bins
    pitch_resampled, pos_data_resampled = {}, {}
    for of in of_keys:
        pitch_resampled[of] = -np.interp(
            bin_centers_sec[of],
            np.arange(len(smooth_pitch_dict[of])) / FS_POS,
            smooth_pitch_dict[of]
        )

        pd_res = {}
        for dim in pos_data_interp[of]:
            pd_res[dim] = np.interp(
                bin_centers_sec[of],
                np.arange(len(pos_data_interp[of][dim])) / FS_POS,
                pos_data_interp[of][dim]
            )
        pos_data_resampled[of] = pd_res

    # Compute and resample speed
    speeds = {}
    for of in of_keys:
        _, _, raw_speed = motive.speed(
            pos_data_interp[of]['X'],
            pos_data_interp[of]['Z'],
            frame_times_all[of]
        )
        speeds[of] = np.interp(
            bin_centers_sec[of],
            np.arange(len(raw_speed)) / FS_POS,
            raw_speed
        )

    # Filter bins by movement/rearing
    spike_matrix_run, pitch_run, pos_data_run, speeds_run = {}, {}, {}, {}
    y_thresh = 0.15
    speed_cutoff = 0.02  # 2 cm/s

    for of in of_keys:
        mask = (pos_data_resampled[of]['Y'] >= y_thresh) | (speeds[of] > speed_cutoff)
        spike_matrix_run[of] = spike_matrix[of][:, mask].T
        pitch_run[of] = pitch_resampled[of][mask]
        speeds_run[of] = speeds[of][mask]

        pos_data_run[of] = {
            dim: pos_data_resampled[of][dim][mask]
            for dim in pos_data_resampled[of]
        }

    return {
        'spike_matrix': spike_matrix,
        'pitch': pitch_resampled,
        'pos_data': pos_data_resampled,
        'speeds': speeds,
        'spike_matrix_run': spike_matrix_run,
        'pitch_run': pitch_run,
        'pos_data_run': pos_data_run,
        'speeds_run': speeds_run,
        'bin_centers_sec': bin_centers_sec,
        'cell_ids': cell_ids,
        'of_keys': of_keys,
        'sl_keys': sl_keys,
        'periods': periods,
        'goodspiketimes': goodspiketimes
    }

# ------------------------------
# Refactor neuronal_classes(), spikes_awake(), spikes_sleep()
# ------------------------------

def neuronal_classes(unit_class_path, unit_list):
    """
    Loads neuronal classification and returns pyr and inter lists.

    Args:
        unit_class_path (Path): Path to the unit_classes_*.pkl file.
        unit_list (list): List of all recorded unit IDs.

    Returns:
        all_units, pyr_units, inter_units
    """
    try:
        with open(unit_class_path, "rb") as f:
            unit_class_dict = pickle.load(f)
        pyr = [u for u in unit_list if unit_class_dict.get(u) == "pyr"]
        inter = [u for u in unit_list if unit_class_dict.get(u) == "int"]
        return unit_list, pyr, inter
    except FileNotFoundError:
        print(f"[Warning] Could not find classification file at {unit_class_path}")
        return unit_list, [], []
    
def spikes_by_periods(goodspiketimes, periods_subset):
    """
    Segments spike times per unit for given subset of periods.

    Args:
        goodspiketimes (dict): All spikes per unit (from oe.ks_load)
        periods_subset (dict): {key: (start, end)} subset of periods (e.g., sl or of only)

    Returns:
        dict of dicts: {period_key: {unit_id: spike_times_within_period}}
    """
    out = {}
    for key, (start, end) in periods_subset.items():
        out[key] = {
            unit: spikes[(spikes >= start) & (spikes <= end)] - start
            for unit, spikes in goodspiketimes.items()
        }
    return out

'''
usage:
Once you've run prepare_umap_inputs(...) and saved the returned dict to data, you can now write:

awake_spikes = spikes_by_periods(data['goodspiketimes'], {k: data['periods'][k] for k in data['of_keys']})
sleep_spikes = spikes_by_periods(data['goodspiketimes'], {k: data['periods'][k] for k in data['sl_keys']})

unit_class_path = Path(res_path) / f"unit_classes_{date}.pkl"
all_units, pyr_units, int_units = neuronal_classes(unit_class_path, list(data['goodspiketimes'].keys()))
'''


# ------------------------------
# PCA & Scaling
# ------------------------------

def run_pca_by_class(
    spike_matrices_dict,
    keys,
    all_units,
    pyr_units=None,
    inter_units=None,
    scale=True,
    n_components=None
):
    """
    Concatenates spike matrices across conditions, optionally scales, and runs PCA.
    Optionally splits by cell type.

    Args:
        spike_matrices_dict: dict of (n_bins, n_cells) arrays
        keys: list of keys to concatenate
        all_units: full list of unit IDs (order = column order of matrices)
        pyr_units / inter_units: lists of unit IDs (subset of all_units)
        scale: whether to z-score each neuron
        n_components: how many PCA dimensions to keep

    Returns:
        dict with keys:
            - pc_all / pc_pyr / pc_inter
            - evr_all / evr_pyr / evr_inter
            - boundaries
    """
    spike_matrices_list = [spike_matrices_dict[k] for k in keys]
    X_all = np.vstack(spike_matrices_list)
    lengths = [m.shape[0] for m in spike_matrices_list]
    boundaries = np.cumsum(lengths)[:-1]

    if scale:
        X_all = StandardScaler().fit_transform(X_all)

    result = {}

    # Masking
    unit_array = np.array(all_units)
    if pyr_units:
        mask_pyr = np.isin(unit_array, pyr_units)
        X_pyr = X_all[:, mask_pyr]
        pca_pyr = PCA(n_components=n_components, random_state=42)
        result['pc_pyr'] = pca_pyr.fit_transform(X_pyr)
        result['evr_pyr'] = pca_pyr.explained_variance_ratio_

    if inter_units:
        mask_inter = np.isin(unit_array, inter_units)
        X_inter = X_all[:, mask_inter]
        pca_inter = PCA(n_components=n_components, random_state=42)
        result['pc_inter'] = pca_inter.fit_transform(X_inter)
        result['evr_inter'] = pca_inter.explained_variance_ratio_

    # PCA on all units
    pca_all = PCA(n_components=n_components, random_state=42)
    result['pc_all'] = pca_all.fit_transform(X_all)
    result['evr_all'] = pca_all.explained_variance_ratio_
    result['boundaries'] = boundaries

    return result

# Optional helper: just get X matrices without PCA

def get_X_by_class(spike_matrices_dict, keys, all_units, pyr_units=None, inter_units=None):
    """
    Returns raw concatenated X matrices (no scaling or PCA), optionally split by class.
    """
    result = {}
    X_all = np.vstack([spike_matrices_dict[k] for k in keys])
    result['X_all'] = X_all
    lengths = [spike_matrices_dict[k].shape[0] for k in keys]
    result['boundaries'] = np.cumsum(lengths)[:-1]

    unit_array = np.array(all_units)
    if pyr_units:
        result['X_pyr'] = X_all[:, np.isin(unit_array, pyr_units)]
    if inter_units:
        result['X_inter'] = X_all[:, np.isin(unit_array, inter_units)]

    return result

'''
Example usage after prepare_umap_inputs()

data = prepare_umap_inputs(...)

all_units, pyr_units, inter_units = neuronal_classes(
    Path(res_path) / f"unit_classes_{date}.pkl",
    list(data['goodspiketimes'].keys())
)

pca_results = run_pca_by_class(
    spike_matrices_dict=data['spike_matrix_run'],
    keys=data['of_keys'],
    all_units=all_units,
    pyr_units=pyr_units,
    inter_units=inter_units,
    scale=True
)

'''
# ------------------------------
# umap
# ------------------------------

def run_umap(
    X,
    umap_kwargs=None,
    split=False,
    boundaries=None
):
    """
    Runs UMAP embedding on input data.

    Args:
        X (array): shape (n_samples, n_features)
        umap_kwargs (dict): UMAP parameters (default reasonable values used if None)
        split (bool): if True, split embedding using boundaries
        boundaries (array): used for splitting embedding into parts

    Returns:
        embedded: either full (n_samples, n_components) or list of segments
    """
    if umap_kwargs is None:
        umap_kwargs = dict(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean',
            random_state=42
        )

    reducer = umap.UMAP(**umap_kwargs)
    emb = reducer.fit_transform(X)

    if split:
        if boundaries is None:
            raise ValueError("If split=True, you must provide boundaries")
        return np.split(emb, boundaries)
    else:
        return emb


'''
flex use
emb_all = run_umap(X, split=False)
emb_split = run_umap(X, split=True, boundaries=boundaries)
'''


# ------------------------------
# plotting
# ------------------------------
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.stats import binned_statistic_2d


def plot_pca_exp_var(evr, save_path=None, title="Explained Variance", threshold=0.90):
    """
    Plots cumulative PCA explained variance.

    Args:
        evr: explained_variance_ratio_ array
        save_path: optional Path to save figure
        title: plot title
        threshold: horizontal line (e.g., 0.9 for 90%)
    """
    cum = np.cumsum(evr)
    plt.figure(figsize=(4, 3))
    plt.plot(np.arange(1, len(cum)+1), cum, marker='o', lw=1)
    plt.axhline(threshold, linestyle='--', color='gray', alpha=0.7)
    plt.xlabel('Number of PCs')
    plt.ylabel('Cumulative explained variance')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# 1. UMAP colored by position + pitch (3-panel)

def plot_umap_by_position(emb, pos_dict, pitch_deg, celltype, of, date, save_path):
    """
    3-panel UMAP colored by X, Z, and pitch.
    """

    from matplotlib.colors import ListedColormap, BoundaryNorm

    BORDERS = [-90,-30, 90]
    LABELS  = ['low', 'high']
    CMAP    = ListedColormap(['#48a0b2', '#fdae61'])  #(['#7B951D', '#FC600A'])
    NORM    = BoundaryNorm(BORDERS, CMAP.N, clip=True)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    sc1 = axs[0].scatter(emb[:,0], emb[:,1], c=pos_dict['X'], cmap='viridis', s=6)
    fig.colorbar(sc1, ax=axs[0], label='X (m)', shrink=0.7, aspect=12)
    axs[0].set_title('Colored by X'); axs[0].set_xlabel('UMAP 1'); axs[0].set_ylabel('UMAP 2')

    sc2 = axs[1].scatter(emb[:,0], emb[:,1], c=pos_dict['Z'], cmap='viridis', s=6)
    fig.colorbar(sc2, ax=axs[1], label='Z (m)', shrink=0.7, aspect=12)
    axs[1].set_title('Colored by Z'); axs[1].set_xlabel('UMAP 1'); axs[1].set_ylabel('UMAP 2')

    pitch = np.asarray(pitch_deg, float)
    valid = np.isfinite(pitch)
    sc3 = axs[2].scatter(emb[valid, 0], emb[valid, 1], c=pitch[valid], cmap=CMAP, norm=NORM, s=6)
    tick_pos = [(-90 + -30)/2, (-30 + 90)/2]
    cbar3 = fig.colorbar(sc3, ax=axs[2], ticks=tick_pos, shrink=0.7, aspect=12)
    cbar3.ax.set_yticklabels(LABELS)
    cbar3.set_label('Pitch (°)')
    axs[2].set_title('Colored by pitch'); axs[2].set_xlabel('UMAP 1'); axs[2].set_ylabel('UMAP 2')

    fig.suptitle(f'UMAP_{celltype}_speedFiltered_{of}_{date}', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path / f'UMAP_{celltype}_speedFiltered_{of}_{date}.png', dpi=300)
    plt.show()

# 2. UMAP colored by scalar across conditions

def plot_umap_by_scalar(ur_emb, arr_run, of_keys, color_variable, celltype, date, save_path):
    """
    Plot UMAPs per condition, colored by any scalar variable (e.g. pitch, speed).
    """
    fig, axs = plt.subplots(1, len(of_keys), figsize=(5 * len(of_keys), 4), constrained_layout=True)

    all_emb = np.concatenate([ur_emb[of] for of in of_keys])
    all_vals = np.concatenate([arr_run[of] for of in of_keys])
    vmin, vmax = np.min(all_vals), np.max(all_vals)

    for ax, of in zip(axs, of_keys):
        emb = ur_emb[of]
        val = arr_run[of]
        sc = ax.scatter(emb[:,0], emb[:,1], c=val, cmap='viridis', s=6, vmin=vmin, vmax=vmax)
        ax.set_title(f'{of}')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')

    fig.colorbar(sc, ax=axs, shrink=0.8, aspect=25).set_label(color_variable)
    plt.suptitle(f'{celltype}_UMAP_colored_by_{color_variable}_{date}', fontsize=14)
    plt.savefig(save_path / f'{celltype}_UMAP_colored_by_{color_variable}_{date}.png', dpi=300)
    plt.show()

# 3. Binned median UMAP by discrete pitch (0 = low, 1 = high)

def plot_umap_binned_median_pitch(ur_emb, pitch_dict, of_keys, celltype, date, save_path, nbins=100,
                                  BORDERS = [-90, -30, 90], LABELS = ['low', 'high']):
    """
    Bins UMAP and computes median pitch bin (low vs high).
    """
    from scipy.stats import binned_statistic_2d
    fig, axs = plt.subplots(1, len(of_keys), figsize=(5 * len(of_keys), 4), constrained_layout=True)

    for ax, of in zip(axs, of_keys):
        emb = ur_emb[of]
        pitch = pitch_dict[of]
        p_binary = np.digitize(pitch, BORDERS) - 1  # 0 = low, 1 = high

        stat, x_edges, y_edges, _ = binned_statistic_2d(
            emb[:, 0], emb[:, 1], p_binary, statistic='median', bins=nbins
        )

        im = ax.imshow(stat.T, origin='lower',
                       extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
                       aspect='auto', cmap='viridis')
        ax.set_title(f'{of}')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')

    fig.colorbar(im, ax=axs, shrink=0.8, aspect=25).set_label('Median pitch (0 = low, 1 = high)')
    plt.suptitle(f'{celltype}_UMAP_binned_by_median_pitch_{date}', fontsize=14)
    plt.savefig(save_path / f'{celltype}_UMAP_binned_by_median_pitch_{date}.png', dpi=300)
    plt.show()


# 4. Grid UMAP of pyr and inter across conditions

def plot_umap_grid_by_class(pyr_dict, inter_dict, conds, date, save_path):
    """
    Plots grid of UMAPs across sessions/conditions for pyr and inter classes.
    """
    def get_global_limits(dicts):
        xs, ys = [], []
        for D in dicts:
            for k in conds:
                if k in D:
                    xs.append(D[k][:,0])
                    ys.append(D[k][:,1])
        x_all = np.concatenate(xs)
        y_all = np.concatenate(ys)
        pad = 0.05
        return (
            x_all.min() - pad * np.ptp(x_all),
            x_all.max() + pad * np.ptp(x_all),
            y_all.min() - pad * np.ptp(y_all),
            y_all.max() + pad * np.ptp(y_all)
        )

    xmin, xmax, ymin, ymax = get_global_limits([pyr_dict, inter_dict])
    fig, axes = plt.subplots(2, len(conds), figsize=(3.2 * len(conds), 6),
                             sharex=True, sharey=True)

    style = dict(s=1, alpha=0.5, linewidths=0)

    for j, cond in enumerate(conds):
        ax_pyr = axes[0, j]
        emb_pyr = pyr_dict.get(cond, None)
        if emb_pyr is not None:
            ax_pyr.scatter(emb_pyr[:,0], emb_pyr[:,1], c='#4C78A8', **style)
        if j == 0:
            ax_pyr.set_ylabel('Pyramidals')
        ax_pyr.set_title(cond)
        ax_pyr.set_xlim(xmin, xmax)
        ax_pyr.set_ylim(ymin, ymax)
        ax_pyr.set_xticks([]); ax_pyr.set_yticks([])
        ax_pyr.set_aspect('equal')

        ax_int = axes[1, j]
        emb_int = inter_dict.get(cond, None)
        if emb_int is not None:
            ax_int.scatter(emb_int[:,0], emb_int[:,1], c='#F58518', **style)
        if j == 0:
            ax_int.set_ylabel('Interneurons')
        ax_int.set_xlim(xmin, xmax)
        ax_int.set_ylim(ymin, ymax)
        ax_int.set_xticks([]); ax_int.set_yticks([])
        ax_int.set_aspect('equal')

    plt.suptitle(f'UMAP_pyr_inter_all_ses_{date}', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path / f'UMAP_pyr_inter_all_ses_{date}.png', dpi=300)
    plt.show()

# ───── UMAP on Spatial / Pitch Rate Maps (Bin-Based, Not Time-Based) ─────

def load_rate_map_dict(res_path, date, of_keys):
    """
    Load 2D spatial rate maps and bin edges for each OF.
    """
    rm_dict = {}
    xy_bins = {}
    for of in of_keys:
        path = Path(res_path) / f"pf/pf_data_dict_{of}_{date}.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        rm_dict[of] = data['rm']
        xy_bins[of] = {'x_edges': data['x_edges'], 'z_edges': data['z_edges']}
    return rm_dict, xy_bins


def load_pitch_map_dict(res_path, date, of_keys):
    """
    Load 1D pitch rate maps for each OF.
    """
    pitch_dict = {}
    for of in of_keys:
        path = Path(res_path) / f"pitch/pf_data_dict1d_{of}_{date}.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        pitch_dict[of] = {'rm': data['rm'], 'pitch_centers': data['pitch_centers']}
    return pitch_dict


def build_population_matrix_2d(fr_dict, units, of_key):
    """
    For spatial rate maps: returns (n_units, n_valid_bins)
    """
    mats = [fr_dict[of_key][u].flatten() for u in units]
    mat = np.vstack(mats)
    valid_mask = ~np.all(np.isnan(mat), axis=0)
    return mat[:, valid_mask], valid_mask


def build_population_matrix_1d(pitch_dict, units, of_key):
    """
    For pitch rate maps: returns (n_units, n_valid_bins)
    """
    mats = [pitch_dict[of_key]['rm'][u].flatten() for u in units]
    mat = np.vstack(mats)
    valid_mask = ~np.all(np.isnan(mat), axis=0)
    return mat[:, valid_mask], valid_mask


def build_across_conditions(build_func, source_dict, units, of_keys):
    """
    Generic function to build population matrix across OFs.
    Returns (full_matrix, [masks], [lengths])
    """
    mats, masks, lengths = [], [], []
    for of in of_keys:
        mat, mask = build_func(source_dict, units, of)
        mats.append(mat)
        masks.append(mask)
        lengths.append(mat.shape[1])
    full_mat = np.hstack(mats)
    return full_mat, masks, lengths

def preprocess_bins(full_mat, scale=True, pca_denoise=False, n_pcs=None):
    """
    Preprocess population matrix before UMAP.
    
    Args:
        full_mat: (n_units, n_bins)
        scale: z-score across bins for each unit
        pca_denoise: whether to run PCA before UMAP
        n_pcs: number of PCs to keep (if None, keep all)

    Returns:
        X_proc: (n_bins, n_features) processed data for UMAP
        pca: fitted PCA object (or None if not used)
    """
    # transpose to (n_bins, n_units)
    X = full_mat.T

    if scale:
        X = StandardScaler().fit_transform(X)

    pca = None
    if pca_denoise:
        pca = PCA(n_components=n_pcs, random_state=42)
        X = pca.fit_transform(X)

    return X, pca

def run_umap_from_bins(full_mat, umap_kwargs=None): # full_mat or X
    """
    Runs UMAP on full population matrix (n_units, n_bins).
    Returns: (n_bins, 2) embedding
    """
    if umap_kwargs is None:
        umap_kwargs = dict(n_neighbors=20, min_dist=0.5, metric='euclidean', random_state=42)

    X = full_mat.T  # UMAP expects (n_samples, n_features) → (n_bins, n_units)
    reducer = umap.UMAP(**umap_kwargs)
    return reducer.fit_transform(X)

def split_embedding(embedding, lengths, of_keys):
    """
    Splits full embedding into per-OF parts using provided lengths.

    Returns:
        dict: {of_key: emb_slice}
    """
    emb_split = {}
    start = 0
    for of, L in zip(of_keys, lengths):
        emb_split[of] = embedding[start:start+L]
        start += L
    return emb_split


def get_flat_bin_coords(xy_bins, of_key):
    """
    Returns flattened (X,Z) bin centers for color mapping.
    """
    x_edges = xy_bins[of_key]['x_edges']
    z_edges = xy_bins[of_key]['z_edges']
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    xx, zz = np.meshgrid(x_centers, z_centers, indexing="xy")
    return xx.flatten(), zz.flatten()


'''
usage

full_mat, masks, lengths = build_across_conditions(build_population_matrix_1d, pitch_dict, pyr_units, ['of1','of2','of3'])
embedding = run_umap_from_bins(full_mat)
embedding_split = split_embedding(embedding, lengths, ['of1','of2','of3'])

# Build population matrix for pyramidal cells in pitch space
pyr_mat, pyr_masks, pyr_lengths = build_across_conditions(build_population_matrix_1d, pitch_dict, pyr_units, ['of1','of2','of3'])

# Preprocess (z-score + PCA)
X_pyr, pca_pyr = preprocess_bins(pyr_mat, scale=True, pca_denoise=True, n_pcs=30)

# Plot explained variance
plot_pca_exp_var_from_bins(pca_pyr, title="Pyramidal pitch bins PCA")

# Run UMAP
emb_pyr = run_umap_from_bins(X_pyr)

# Split into OFs
emb_pyr_split = split_embedding(emb_pyr, pyr_lengths, ['of1','of2','of3'])
'''
# plotting
def plot_pca_exp_var_from_bins(pca, save_path=None, title="Explained Variance", threshold=0.90):
    """
    Plot cumulative explained variance from PCA object.
    """
    if pca is None:
        print("[Info] No PCA was applied, skipping EVR plot.")
        return

    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)

    plt.figure(figsize=(4, 3))
    plt.plot(np.arange(1, len(cum)+1), cum, marker='o', lw=1)
    plt.axhline(threshold, linestyle='--', color='gray', alpha=0.7)
    plt.xlabel('Number of PCs')
    plt.ylabel('Cumulative explained variance')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_umap_colored_by_scalar(emb_split, scalar_dict, of_keys, title="", cmap="viridis", save_path=None):
    """
    Plots UMAP per OF, colored by scalar (pitch, X, Z, etc.).

    Args:
        emb_split: dict {of: (n_bins, 2)} UMAP embedding
        scalar_dict: dict {of: array of scalars for each bin}
        of_keys: list of OFs
        title: suptitle
    """
    fig, axs = plt.subplots(1, len(of_keys), figsize=(5*len(of_keys), 4), constrained_layout=True)

    all_vals = np.concatenate([scalar_dict[of] for of in of_keys])
    vmin, vmax = np.min(all_vals), np.max(all_vals)

    for ax, of in zip(axs, of_keys):
        sc = ax.scatter(emb_split[of][:,0], emb_split[of][:,1], 
                        c=scalar_dict[of], cmap=cmap, vmin=vmin, vmax=vmax, s=10)
        ax.set_title(of)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

    cbar = fig.colorbar(sc, ax=axs, shrink=0.7, aspect=25)
    cbar.set_label(title)

    plt.suptitle(title, fontsize=14)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_umap_grid_by_class_bins(emb_pyr_split, emb_int_split, of_keys, date, save_path=None):
    """
    Grid plot of UMAP embeddings for pyramidal vs interneurons across OFs.
    Rows = cell class, Cols = OFs
    """
    fig, axes = plt.subplots(2, len(of_keys), figsize=(3.2*len(of_keys), 6),
                             sharex=True, sharey=True)

    style = dict(s=5, alpha=0.7, linewidths=0)

    for j, of in enumerate(of_keys):
        # Row 0: Pyr
        ax = axes[0, j]
        if of in emb_pyr_split:
            ax.scatter(emb_pyr_split[of][:,0], emb_pyr_split[of][:,1], c="#4C78A8", **style)
        if j == 0:
            ax.set_ylabel("Pyramidal")
        ax.set_title(of)
        ax.set_aspect("equal")

        # Row 1: Inter
        ax = axes[1, j]
        if of in emb_int_split:
            ax.scatter(emb_int_split[of][:,0], emb_int_split[of][:,1], c="#F58518", **style)
        if j == 0:
            ax.set_ylabel("Interneuron")
        ax.set_aspect("equal")

    plt.suptitle(f"UMAP on bin-based rate maps ({date})", y=0.98)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_umap_overlay(embedding, lengths, of_keys, title="", save_path=None):
    """
    Overlays all OF embeddings in one plot with color-coded OFs.
    """
    start = 0
    colors = plt.cm.tab10.colors
    plt.figure(figsize=(5,4))
    for of, L, col in zip(of_keys, lengths, colors):
        plt.scatter(embedding[start:start+L, 0], embedding[start:start+L, 1], 
                    c=[col], label=of, s=10, alpha=0.8)
        start += L
    plt.legend()
    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

'''
# Build + preprocess pyr pitch matrix
pyr_mat, pyr_masks, pyr_lengths = build_across_conditions(build_population_matrix_1d, pitch_dict, pyr_units, ['of1','of2','of3'])
X_pyr, pca_pyr = preprocess_bins(pyr_mat, scale=True, pca_denoise=True, n_pcs=20)
emb_pyr = run_umap_from_bins(X_pyr)
emb_pyr_split = split_embedding(emb_pyr, pyr_lengths, ['of1','of2','of3'])

# Same for interneurons
int_mat, int_masks, int_lengths = build_across_conditions(build_population_matrix_1d, pitch_dict, int_units, ['of1','of2','of3'])
X_int, pca_int = preprocess_bins(int_mat, scale=True, pca_denoise=True, n_pcs=20)
emb_int = run_umap_from_bins(X_int)
emb_int_split = split_embedding(emb_int, int_lengths, ['of1','of2','of3'])

# Plot grid
plot_umap_grid_by_class_bins(emb_pyr_split, emb_int_split, ['of1','of2','of3'], date, save_path=Path("pyr_int_grid.png"))

# Plot colored by pitch bins
pyr_pitch_colors = {of: pitch_dict[of]['pitch_centers'][mask] for of, mask in zip(['of1','of2','of3'], pyr_masks)}
plot_umap_colored_by_scalar(emb_pyr_split, pyr_pitch_colors, ['of1','of2','of3'], title="Pyramidal pitch bins")

'''






    
















