from .matlab_interface import loadmat


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

        self.abs_engaged_observ_start_times  = from_dict['engaged_observ_start_times']
        self.abs_engaged_observ_end_times    = from_dict['engaged_observ_end_times']

        self.abs_grooming_observ_start_times = from_dict['grooming_observ_start_times']
        self.abs_grooming_observ_end_times   = from_dict['grooming_observ_end_times']

        self.abs_resting_observ_start_times  = from_dict['resting_observ_start_times']
        self.abs_resting_observ_end_times    = from_dict['resting_observ_end_times']

        self.x1 = from_dict['x1']
        self.y1 = from_dict['y1']

        if isinstance(self.recorded_cells, float) or isinstance(self.recorded_cells, int):
            self.recorded_cells = [self.recorded_cells]
        if isinstance(self.recorded_cells_id, float) or isinstance(self.recorded_cells_id, int):
            self.recorded_cells_id = [self.recorded_cells_id]


    def __repr__(self):
        return str(self.__dict__)

