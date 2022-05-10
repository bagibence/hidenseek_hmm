import numpy as np


def find_nearest_idx(array, value, threshold=100):
    """
    Find index of the element closest to a value in an array
    
    Parameters
    ----------
    array : array-like
        array to search in
    value : float
        value to search for
    threshold : float
        throw an exception if the difference between the value
        and the closest element is bigger than threshold
        
    Returns
    -------
    idx : int
        index of the element that is closest to value in array
    """
    array = np.asarray(array)
    diffs = np.abs(array - value)
    
    if diffs.min() < threshold:
        return diffs.argmin()
    else:
        raise Exception('no frame close enough')


def frame_to_time(i, fps):
    """
    What time the i-th frame in a video with a given fps appears
    
    Parameters
    ----------
    i : int
        index of the frame
    fps : float
        fps of the video
        
    Return
    ------
    t = i / fps
    """
    return i / fps


def neural_to_video_time(timepoint, frametimes, fps):
    """
    Find the timestamp in the video corresponding to a timepoint in the neural data
    
    Parameters
    ----------
    timepoints : float
        time point in the neural data's time coordinates
    frametimes : array
        times of the video's frames in the neural data's time coordinates
    fps : float
        fps of the video
        
    Returns
    -------
    timepoint in the video's time coordinates
    """
    frame_ind = find_nearest_idx(frametimes, timepoint)
    return frame_to_time(frame_ind, fps)
