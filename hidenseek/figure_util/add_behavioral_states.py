from hidenseek.db_interface import *

from hidenseek.globals import (
        behavioral_state_names_seek,
        behavioral_state_names_hide,
        behavioral_state_names_extra,
        behavioral_state_dict,
        inverse_behavioral_state_dict
)

from hidenseek.util.postproc import make_behavioral_states_int, make_behavioral_states_str, convert_str_states_to_int

import xarray as xr
from tqdm.auto import tqdm


def add_behavioral_states():
    for session in tqdm(Session.select()):
        for trial in session.trials:
            trial.behavioral_states_str = make_behavioral_states_str(trial, extra_states = True, combine_observing_states = True)
            trial.behavioral_states = convert_str_states_to_int(trial.behavioral_states_str, behavioral_state_dict)

        session.behavioral_states = xr.concat([t.behavioral_states.drop('time') for t in session.trials], dim = 'time')


def add_playing_observing_states():
    for session in tqdm(Session.select()):
        for trial in session.trials:
            trial.playing_states_str = make_behavioral_states_str(trial, extra_states = True, observing_states=False)
            trial.playing_states = make_behavioral_states_int(trial, behavioral_state_dict, extra_states = True, observing_states = False)
            
            if trial.observing:
                trial.observing_states_str = make_behavioral_states_str(trial, playing_states = False)
                trial.observing_states = make_behavioral_states_int(trial, behavioral_state_dict, playing_states = False)
