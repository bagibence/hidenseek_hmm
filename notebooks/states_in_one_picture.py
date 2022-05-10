# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python (hidenseek)
#     language: python
#     name: hidenseek
# ---

# %% [markdown]
# This notebooks combines video frames into a single picture for a given state.
#
# The video files are not online, but the code might be useful.

# %%
import pickle
import os

import numpy as np
import pandas as pd
import xarray as xr
import scipy.signal as scs
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import seaborn as sns
# %matplotlib inline

from tqdm.auto import tqdm

# %%
from moviepy.editor import *
import proglog
proglog.notebook()

# %%
from hidenseek.db_interface import *
connect_to_db(os.path.join(os.getenv('INTERIM_DATA_DIR'), 'database.db'))

from hidenseek.figure_util.load_results import load_results
from hidenseek.figure_util.video import *

# %% [markdown]
# # Load results 

# %%
K = 11
transitions = 'sticky'
n_seeds = 40

load_results(K, transitions, n_seeds=n_seeds)

# %%
figures_root_dir = os.path.join(os.getenv('ROOT_DIR'), 'reports', 'figures', 'states_in_one_image')
figures_dir = os.path.join(figures_root_dir, f'GPFA_{K}_states_{n_seeds}_seeds')

for d in [figures_root_dir, figures_dir]:
    if not os.path.exists(d):
        os.mkdir(d)
# %% [markdown]
# # Session and states 

# %%
session = Session[12]

states_to_do = [9]

# %% [markdown]
# # Get the videos

# %%
session_dir = os.path.join(figures_dir, f"session_{session.id}")

if not os.path.exists(session_dir):
    os.mkdir(session_dir)

# %%
video_dir = f'/Volumes/Maxtor/HideNSeek/{session.animal}/{session.date}/Bonsai'

video_filename1 = [fname for fname in  os.listdir(video_dir) if fname.startswith("VideoCam1") and fname.endswith('.avi')][0]
video_filename2 = [fname for fname in  os.listdir(video_dir) if fname.startswith("VideoCam2") and fname.endswith('.avi')][0]

clip1 = VideoFileClip(os.path.join(video_dir, video_filename1))
clip2 = VideoFileClip(os.path.join(video_dir, video_filename2))

# %%
# run this once to know the sizes beforehand

# in session 13 the rat is in the first frame but not in the saved pictures
if session.id == 13:
    im1 = np.rot90(plt.imread(os.path.join(video_dir, "ImageCam1.png")), 2)
    im2 = np.rot90(plt.imread(os.path.join(video_dir, "ImageCam2.png")), 2)
    arena = np.rot90(np.vstack([im1, im2]), 2)
# can't use the saved pictures because the layout is changed in different sessions
else:
    im1 = clip1.get_frame(0)
    im2 = clip2.get_frame(0)
    arena = np.vstack([im2, im1])


# %%
def get_state_frames(k, downsample_rate):
    """
    Get frames from the videos when a state was active
    
    Parameters
    ----------
    k : int
        state ID
    downsample_rate : int
        get every n-th frame
        
    Returns
    -------
    list of frames as np.arrays
    """
    state_frames = []
    for trial in tqdm(session.trials):
        state_times = trial.states.time.values[trial.states.values == k] + trial.abs_time_points.start

        for t in state_times[::downsample_rate]:
            t1 = neural_to_video_time(t, session.frametimes1, clip1.fps)
            t2 = neural_to_video_time(t, session.frametimes2, clip2.fps)

            # get the frame from the two videos
            frame1 = clip1.get_frame(t1)
            frame2 = clip2.get_frame(t2)

            # combine them the same way as the arena
            whole_frame = np.vstack([frame2, frame1])

            state_frames.append(whole_frame)
    
    return state_frames


# %%
def reload_arena():
    # in session 13 the rat is in the first frame but not in the saved pictures
    if session.id == 13:
        im1 = np.rot90(plt.imread(os.path.join(video_dir, "ImageCam1.png")), 2)
        im2 = np.rot90(plt.imread(os.path.join(video_dir, "ImageCam2.png")), 2)
        arena = np.rot90(np.vstack([im1, im2]), 2)
    # can't use the saved pictures because the layout is changed in different sessions
    else:
        im1 = clip1.get_frame(0)
        im2 = clip2.get_frame(0)
        arena = np.vstack([im2, im1]) / 255
    
    return arena


# %%
arena = reload_arena()

# %% [markdown]
# # Filter or not 

# %%
filter_invalid_pos = True

# %% [markdown]
# # Combine images based on the positions 

# %%
every_nth_frame = 2

# %%
by_pos_dir = os.path.join(session_dir, "by_position")

if not os.path.exists(by_pos_dir):
    os.mkdir(by_pos_dir)


# %%
def bounding_box(x, y, picshape, size=100):
    """
    Bounding box around (x, y) without indexing out of the image
    
    Parameters
    ----------
    x, y : int
        center of the box
    picshape : tuple of length 3
        y_shape, x_shape, depth
        given by arena.shape
    size : int, default 100
        size of the bounding box
        
    Returns
    -------
    tuple of slices for indexing the source image
    """
    ymax, xmax, _ = picshape
    s = size // 2

    x0 = max(x - s, 0)
    x1 = min(x + s, xmax)
    y0 = max(y - s, 0)
    y1 = min(y + s, ymax)

    return slice(y0, y1), slice(x0, x1)


# %% [markdown]
# ## every n-th frame 

# %% tags=[]
bounding_size = 100

for k in tqdm(states_to_do):
    arena = reload_arena()

    for trial in tqdm(session.trials):
        state_times = trial.states.time.values[trial.states.values == k] + trial.abs_time_points.start

        pos2 = session.pos2.sel(time = state_times, method='nearest').copy(deep=True)

        pos1 = session.pos1.sel(time = state_times, method = 'nearest').copy(deep=True)
        pos1['y'] = pos1['y'] + im1.shape[0] # shift so that they can be plotted into a single image

        if filter_invalid_pos:
            pos2['x'][pos2.x.values < 140] = np.nan
            pos2['y'][pos2.x.values < 140] = np.nan

        for t in state_times[::every_nth_frame]:
            # translate the time to video time
            t1 = neural_to_video_time(t, session.frametimes1, clip1.fps)
            t2 = neural_to_video_time(t, session.frametimes2, clip2.fps)

            # get the frame from the two videos
            frame1 = clip1.get_frame(t1)
            frame2 = clip2.get_frame(t2)

            # combine them the same way as the arena
            whole_frame = np.vstack([frame2, frame1])

            for pos in [pos1, pos2]:
                # get the position in integers
                x0 = float(pos.x.sel(time=t, method='nearest'))
                y0 = float(pos.y.sel(time=t, method='nearest'))

                if not np.all(np.isfinite([x0, y0])):
                    continue

                x0 = int(round(x0))
                y0 = int(round(y0))

                # overwrite the arena pic
                # images are indexed differently
                arena[bounding_box(x0, y0, arena.shape, bounding_size)] = whole_frame[bounding_box(x0, y0, arena.shape, bounding_size)] / 255


    fig, ax = plt.subplots(figsize = (20, 30))
    ax.imshow(arena)

    fig.savefig(os.path.join(by_pos_dir, f"state_{k}_every_{every_nth_frame}_frame.png"), dpi=300)
    np.savez(os.path.join(by_pos_dir, f"state_{k}_every_{every_nth_frame}_frame.npz"), im=arena)

    plt.close()

# %% [markdown]
# # Using cv2 

# %%
every_nth_frame = 2
pixel_ds_rate = 1

# %%
by_cont_dir = os.path.join(session_dir, "by_cont")

if not os.path.exists(by_cont_dir):
    os.mkdir(by_cont_dir)

# %%
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity
from skimage.color import rgb2gray
import cv2

def get_diff_mask(arena, frame, area_thresh):
    """
    Find areas that are different between the background and the current frame
    
    Adapted from the top answer at https://stackoverflow.com/questions/56183201/detect-and-visualize-differences-between-two-images-with-opencv-python
    Direct link: https://stackoverflow.com/a/56193442
    
    Parameters
    ----------
    arena : 3D np.array
        background image
    frame : 3D np.array
        current frame
    area_thresh : float
        minimum patch area
        smaller different patches are not considered
        used to filter noise
    
    Returns
    -------
    boolean 2D mask showing differences
    """
    im_a = rgb2gray(arena / 255)
    im_b = rgb2gray(frame / 255)
    
    score, diff = structural_similarity(im_a, im_b, full=True)
    diff = (diff * 255).astype("uint8")
    
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    mask = np.zeros(arena.shape, dtype='uint8')
    for c in contours:
        area = cv2.contourArea(c)
        if area > area_thresh:
            cv2.drawContours(mask, [c], 0, (0,255,0), -1)

    return mask[:, :, 1] == 255


from hidenseek.util.postproc.transitions import get_transition_state_pairs, get_transition_points
from hidenseek.util.postproc.misc import correct_starts_and_ends

def get_frame_from_every_occurrence(session, k):
    """
    Get a single (middle) frame from every segment a state becomes active
    
    Parameters
    ----------
    session : Session
        session to consider
    k : int
        state ID
    """
    state_frames = []
    for trial in tqdm(session.trials):
        pairs = get_transition_state_pairs(trial.states)
        transition_times = get_transition_points(trial.states)

        starts, ends = [], []
        for p, t in zip(pairs, transition_times):
            if p[1] == k:
                starts.append(t)
            if p[0] == k:
                ends.append(t)
        starts, ends = correct_starts_and_ends(starts, ends)
        
        abs_middles = []
        for (s, e) in zip(starts, ends):
            if (s is not None) and (e is not None):
                abs_middles.append((s + e) / 2 + trial.abs_time_points.start)

        middles1 = [neural_to_video_time(t, session.frametimes1, clip1.fps) for t in abs_middles]
        middles2 = [neural_to_video_time(t, session.frametimes2, clip2.fps) for t in abs_middles]
        
        for (t1, t2) in zip(middles1, middles2):
            # get the frame from the two videos
            frame1 = clip1.get_frame(t1)
            frame2 = clip2.get_frame(t2)

            # combine them the same way as the arena
            whole_frame = np.vstack([frame2, frame1])

            state_frames.append(whole_frame)
            
    return state_frames

def combine_state_frames_cont(state_frames, pixel_ds_rate, area_thresh=40):
    """
    Overlay interesting parts of the frames on the background image
    
    Parameters
    ----------
    state_frames : list of np.array
        list of frames to combine
    pixel_ds_rate : int
        downsample images for faster processing
    area_thresh : float, default 40
        minimum patch area
        passed to get_diff_mask
    
    Returns
    -------
    combined image as an np.array
    """
    arena = reload_arena() * 255
    arena = arena[::pixel_ds_rate, ::pixel_ds_rate, :]
    orig_arena = arena.copy()
    state_frames = [frame[::pixel_ds_rate, ::pixel_ds_rate, :] for frame in state_frames]
    
    for frame in tqdm(state_frames):
        frame_mask = get_diff_mask(orig_arena, frame, area_thresh)

        if np.any(frame_mask):
            arena[frame_mask] = frame[frame_mask]
            
    return arena


# %% [markdown]
# ## Every n-th frame 

# %% tags=[]
for k in tqdm(states_to_do):
    state_frames = get_state_frames(k, every_nth_frame)
    
    combined_image = combine_state_frames_cont(state_frames, pixel_ds_rate)

    fig, ax = plt.subplots(figsize = (20, 30))
    ax.imshow(combined_image / 255)
    
    fig.savefig(os.path.join(by_cont_dir, f"state_{k}_every_{every_nth_frame}_frame.png"), dpi=300)
    np.savez(os.path.join(by_cont_dir, f"state_{k}_every_{every_nth_frame}_frame.npz"), im=combined_image)
    
    plt.close()

# %% [markdown]
# ## Middle frame from every occurrence 

# %%
every_occ_dir = os.path.join(session_dir, "every_occurrence_by_cont")
if not os.path.exists(every_occ_dir):
    os.mkdir(every_occ_dir)
    
for k in tqdm(states_to_do):
    state_frames = get_frame_from_every_occurrence(session, k)
    
    combined_image = combine_state_frames_cont(state_frames, pixel_ds_rate)

    fig, ax = plt.subplots(figsize = (20, 30))
    ax.imshow(combined_image / 255)
    
    fig.savefig(os.path.join(every_occ_dir, f"state_{k}.png"), dpi=300)
    np.savez(os.path.join(every_occ_dir, f"state_{k}.npz"), im=combined_image)
    
    plt.close()

# %% [markdown]
# # Combine frames based on deviation from the average frame 

# %%
every_nth_frame = 2
pixel_ds_rate = 4

# %%
by_std_dir = os.path.join(session_dir, "by_std")

if not os.path.exists(by_std_dir):
    os.mkdir(by_std_dir)

# %% tags=[]
n_std = 3

for k in tqdm(states_to_d):
    state_frames = get_state_frames(k, every_nth_frame)
    
    #state_frames = [gaussian_filter(frame, 5) for frame in state_frames]
    
    state_frames = [frame[::pixel_ds_rate, ::pixel_ds_rate, :] for frame in state_frames]
    
    state_frames_arr = np.stack(state_frames)
    del state_frames
    
    # loop through pixels to not run out of memory
    #av_frame = np.zeros_like(state_frames_arr[0])
    #frame_std = np.zeros_like(state_frames_arr[0])
    #x, y, _ = state_frames_arr[0].shape
    #for i in tqdm(range(x)):
    #    for j in range(y):
    #        av_frame[i, j, :] = np.mean(state_frames_arr[:, i, j, :], axis = 0)
    #        frame_std[i, j, :] = np.std(state_frames_arr[:, i, j, :], axis = 0)
            
    av_frame = np.mean(state_frames_arr, axis=0)
    frame_std = np.std(state_frames_arr, axis=0)

    arena = reload_arena() * 255
    arena = arena[::pixel_ds_rate, ::pixel_ds_rate, :]

    for frame in tqdm(state_frames_arr):
        frame_mask = ((np.abs(frame - av_frame) > n_std*frame_std).sum(axis=2) > 0)
        
        if np.any(frame_mask):
            arena[frame_mask] = frame[frame_mask]

    fig, ax = plt.subplots(figsize = (20, 30))
    ax.imshow(arena / 255)
    
    fig.savefig(os.path.join(by_std_dir, f"state_{k}_every_{every_nth_frame}_frame_{n_std}_std.png"), dpi=300)
    np.savez(os.path.join(by_std_dir, f"state_{k}_every_{every_nth_frame}_frame_{n_std}_std.npz"), im=arena.astype(int))
    
    plt.close()

# %%
