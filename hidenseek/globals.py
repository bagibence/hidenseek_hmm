"""
Useful global variables for the hidenseek project
"""

import numpy as np


time_point_names_bo = ['start', 'box_open', 'jump_out', 'interaction', 'transit', 'jump_in', 'end']
time_point_names = ['start', 'jump_out', 'interaction', 'transit', 'jump_in', 'end']

time_point_names_bo_plot = ['start', 'box open', 'jump out', 'interaction', 'transit', 'jump in', 'end']
time_point_names_plot = ['start', 'jump out', 'interaction', 'transit', 'jump in', 'end']


behavioral_state_names_seek = ['box_start_closed', 'box_start_open', 'game_seek', 'interaction', 'transit', 'box_end_open']
behavioral_state_names_hide = ['box_start_open', 'game_hide', 'interaction', 'transit', 'box_end_open']
behavioral_state_names_extra = ['darting', 'exploring', 'hiding', 'engaged_observing', 'grooming_observing', 'resting_observing', 'random_observing']

# unique returns sorted (aka consistent) while set is random I think
behavioral_state_names_all = list(np.unique(behavioral_state_names_seek + behavioral_state_names_hide + behavioral_state_names_extra))

# start at 100 to be distinct from the HMM states
behavioral_state_dict = {name : i for i, name in enumerate(behavioral_state_names_all, 100)}
inverse_behavioral_state_dict = {v : k for k, v in behavioral_state_dict.items()}

# pretty names of the behavioral states
behavioral_state_plot_dict = {'box_end_open'     : 'In open box (end)',
                              'box_start_closed' : 'In closed box (start)',
                              'box_start_open'   : 'In open box (start)',
                              'darting'          : 'Darting',
                              'exploring'        : 'Exploring',
                              'game_hide'        : 'Misc. in hide',
                              'game_seek'        : 'Misc. in seek',
                              'hiding'           : 'Hiding',
                              'interaction'      : 'Interaction',
                              'transit'          : 'Transit'}
