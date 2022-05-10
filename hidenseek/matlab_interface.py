class mCell:
    def __init__(self, from_dict):
        self.new_id      = from_dict['new_id']
        self.cluster_id  = from_dict['new_id']
        self.animal      = from_dict['animal']
        self.date        = from_dict['date']
        self.session_id  = from_dict['session_id']
        self.all_spikes  = from_dict['all_spikes']

    def __repr__(self):
        return str(self.__dict__)


class mSession:
    def __init__(self, from_dict):
        self.animal            = from_dict['animal']
        self.date              = from_dict['date']
        self.session_id        = from_dict['session_id']
        self.recorded_cells    = from_dict['recorded_cells']
        self.recorded_cells_id = from_dict['recorded_cells_id']
        self.all_time_points   = from_dict['all_time_points']

        self.seek_trial_start_times      = from_dict['seek_trial_start_times']
        self.seek_trial_end_times        = from_dict['seek_trial_end_times']

        self.hide_trial_start_times      = from_dict['hide_trial_start_times']
        self.hide_trial_end_times        = from_dict['hide_trial_end_times']

        self.abs_darting_start_times     = from_dict['darting_start_times']
        self.abs_darting_end_times       = from_dict['darting_end_times']

        self.abs_interaction_start_times = from_dict['interaction_start_times']
        self.abs_interaction_end_times   = from_dict['interaction_end_times']

        self.abs_jumpin_times            = from_dict['jumpin_times']
        self.abs_jumpout_times           = from_dict['jumpout_times']

        self.abs_box_open_times          = from_dict['box_open_times']
        self.abs_box_closed_times        = from_dict['box_closed_times']

        self.abs_transit_start_times     = from_dict['transit_start_times']
        self.abs_transit_end_times       = from_dict['transit_end_times']

        self.abs_sighting_times          = from_dict['sighting_times']

        self.abs_exploring_start_times   = from_dict['exploring_start_times']
        self.abs_exploring_end_times     = from_dict['exploring_end_times']

        self.abs_hiding_start_times      = from_dict['hiding_start_times']
        self.abs_hiding_end_times        = from_dict['hiding_end_times']

        self.frametimes1 = from_dict['frametimes1']
        self.frametimes2 = from_dict['frametimes2']

        self.x1 = from_dict['x1']
        self.y1 = from_dict['y1']
        self.x2 = from_dict['x2']
        self.y2 = from_dict['y2']

        self.call_times = from_dict['call_times']
        self.call_types = from_dict['call_types']

        if isinstance(self.recorded_cells, float) or isinstance(self.recorded_cells, int):
            self.recorded_cells = [self.recorded_cells]
        if isinstance(self.recorded_cells_id, float) or isinstance(self.recorded_cells_id, int):
            self.recorded_cells_id = [self.recorded_cells_id]


    def __repr__(self):
        return str(self.__dict__)



import numpy as np
import scipy.io
def loadmat(filename):
    '''
    this function should be called instead of direct scipy.io.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    
    From https://blog.nephics.se/2019/08/28/better-loadmat-for-scipy/
    ... which looks like is part of scipy since then...
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], scipy.io.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
            elif isinstance(d[key], np.ndarray):
            #elif _has_struct(d[key]):
                d[key] = _tolist(d[key])
        return d

    def _has_struct(elem):
        """Determine if elem is an array and if any array item is a struct"""
        return isinstance(elem, np.ndarray) and any(isinstance(
                    e, scipy.io.matlab.mio5_params.mat_struct) for e in elem)

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif _has_struct(elem):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, scipy.io.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif _has_struct(sub_elem):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

