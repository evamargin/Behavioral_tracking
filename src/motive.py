import pandas as pd
import numpy as np
import csv
from scipy.spatial.transform import Rotation as R
import os
import glob
import shutil
from pathlib import Path

def get_csv_dict(csv_folder_path):

    '''
    returns dict with df for each session
    keys - session names like ['20250718_1','20250718_2',...]
    '''

    csv_dict = {}

    csv_folder_path = Path(csv_folder_path)
    for file_csv in csv_folder_path.glob("*.csv"):
        key = file_csv.stem
        rows = []
        with open(file_csv, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                if len(line) > 1:  # skip empty or malformed lines
                    rows.append(line)
        try:
            df = pd.DataFrame(rows)
            csv_dict[key] = df
        except Exception as e:
            print(f"Failed to convert {file_csv.name}: {e}")  

    return csv_dict

def get_meta(df):

    '''
    U ll get this dict:

    {'Format Version': '1.21',
    'Take Name': 'Take 2025-07-18 01.41.56 PM',
    'Capture Frame Rate': '120.000046',
    'Export Frame Rate': '120.000046',
    'Capture Start Time': '2025-07-18 01.41.56.441 PM',
    'Total Frames in Take': '220314',
    'Total Exported Frames': '220314',
    'Rotation Type': 'Quaternion',
    'Length Units': 'Meters',
    'Coordinate Space': 'Global',
    'Duration': 1835.940966}
    '''

    motive_meta = df.iloc[0, 0:20].tolist()
    motive_meta = {motive_meta[i]: motive_meta[i + 1] for i in range(0, len(motive_meta) - 1, 2)}
    trial_duration = float(df.iloc[-1, 1])
    motive_meta['Duration'] = trial_duration # sec

    return motive_meta 

def get_array(df, rb='rat_v2', metric='Position', dim='X'):
    '''
    slice from df any column u need
    returns as np array
    '''
    try:
        mask = (
            (df.iloc[2] == rb) &
            (df.iloc[4] == metric) &
            (df.iloc[5] == dim)
        )
        col_slice = df[df.columns[mask]]
        col_slice = col_slice.iloc[6:,0]
        col_slice = col_slice.to_numpy()
        col_slice = np.where(col_slice == '', np.nan, col_slice).astype(float)
    except:
        mask = (
            (df.iloc[2+1] == rb) &
            (df.iloc[4+1] == metric) &
            (df.iloc[5+1] == dim)
        )
        col_slice = df[df.columns[mask]]
        col_slice = col_slice.iloc[6+1:,0]
        col_slice = col_slice.to_numpy()
        col_slice = np.where(col_slice == '', np.nan, col_slice).astype(float)

    #print(dim, ':', np.isnan(col_slice).sum(), 'lost frames')
    
    return col_slice

def get_arrays(df, metric='Position', dim_array = ['X','Y','Z'], interpolate=True):

    # Determine if we need an offset in row indexing
    offset = 0 if (df.iloc[1] == 'Rigid Body').any() else 1

    # Get indices of rigid bodies
    idx = np.where(df.iloc[1 + offset] == 'Rigid Body')[0]

    # Get unique rigid body names
    rbs = pd.unique(df.iloc[2 + offset, idx])  # e.g., ['TOF', 'rat_v2']
        
    i_rat = np.where(np.char.find(rbs.astype(str), 'rat') != -1)[0] # where rb name has rat in the name
    rb = rbs[i_rat][0]

    # get centroid position/angle
    arrays = {}
    for dim in dim_array:
        arrays[dim] = get_array(df, rb, metric, dim)

    arrays_interpol = {}
    if interpolate:
        for dim in dim_array:
            rat_arr = arrays[dim]
            arrays_interpol[dim] = pd.Series(rat_arr).interpolate(method='linear', limit_direction='both').to_numpy()

    return arrays, arrays_interpol

def get_frame_times(df):
    frame_times = df.iloc[6:, 1]
    frame_times = pd.to_numeric(frame_times, errors='coerce').dropna()
    frame_times = frame_times.to_numpy()

    return frame_times



def speed(x_interp, z_interp, frame_times, smooth_s = 0.5, motive_sr=120):
    '''
    smooth_s: if u wanna smooth for 500ms --> pick 0.5s
    u can calculate sigma (in samples): motive_sr=samples/sec --> samples = 120*sec
    '''
    from scipy.ndimage import gaussian_filter1d

    dx = np.diff(x_interp)
    dz = np.diff(z_interp)
    dt = np.diff(frame_times)

    v_xz = np.sqrt(dx**2 + dz**2) / dt

    # Append last value to keep same length with positional data
    speed = np.append(v_xz, v_xz[-1])  # shape = (n_frames,)

    # smooth speed to remove jitter
    sigma = int(motive_sr*smooth_s)
    smoothed_speed = gaussian_filter1d(speed, sigma=sigma)

    return v_xz, speed, smoothed_speed





    






