from pony.orm import *

import numpy as np
import xarray as xr

import os
import warnings
from cached_property import cached_property

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv('config.dotenv'))

from hidenseek.db_mixins import TrialMixin, SessionMixin
import hidenseek.preprocessing as preproc

db = Database()


class Cell(db.Entity):
    id          = PrimaryKey(int)
    cluster_id  = Required(int)
    session     = Required('Session')
    _all_spikes = Required(FloatArray)

    # session and cluster id together must be unique
    composite_key(session, cluster_id)


    @property
    def all_spikes(self):
        """All spiketimes as np.array"""
        return np.array(self._all_spikes)


    @property
    def trials(self):
        return self.session.trials


    @property
    def seek_trials(self):
        return self.session.seek_trials


    @property
    def hide_trials(self):
        return self.session.hide_trials


class Trial(db.Entity, TrialMixin):
    role                = Required(str)
    session             = Required('Session')
    abs_start_time      = Required(float)
    abs_end_time        = Required(float)


    @property
    def recorded_cells(self):
        return self.session.recorded_cells


    @property
    def abs_spike_times(self):
        """
        List of spike times of all cells recorded during this trial
        filtered so that only spikes in this trial are kept
        """
        return [self.filter(c.all_spikes, inclusive=True)
                for c in self.recorded_cells]


    @property
    def spike_times(self):
        """
        List of spike times of all cells recorded during this trial
        filtered so that only spikes in this trial are kept
        and shifted so that they are relative times
        from the start of the trial
        """
        return [neur_times - self.abs_start_time for neur_times in self.abs_spike_times]


    @cached_property
    def observing(self):
        return any([(start <= self.abs_time_points.start <= end) for start, end in zip(self.session.abs_observing_start_times, self.session.abs_observing_end_times)]) or (len(self.filter(self.session.abs_observing_times)) > 0)


    @property
    def observing_role(self):
        if self.observing:
            return 'observing'
        else:
            return 'playing'


class Session(db.Entity, SessionMixin):
    id             = PrimaryKey(int)
    animal         = Required(str)
    date           = Required(str)
    _recorded_cells = Set('Cell')
    _trials         = Set('Trial')

    paper_id = Optional(int)

    _all_time_points = Required(FloatArray)

    abs_darting_start_times     = Required(FloatArray)
    abs_darting_end_times       = Required(FloatArray)

    abs_interaction_start_times = Required(FloatArray)
    abs_interaction_end_times   = Required(FloatArray)

    abs_jump_in_times           = Required(FloatArray)
    abs_jump_out_times          = Required(FloatArray)

    abs_box_open_times          = Required(FloatArray)
    abs_box_closed_times        = Required(FloatArray)

    abs_transit_start_times     = Required(FloatArray)
    abs_transit_end_times       = Required(FloatArray)

    abs_sighting_times          = Required(FloatArray)

    abs_exploring_start_times   = Required(FloatArray)
    abs_exploring_end_times     = Required(FloatArray)

    abs_hiding_start_times      = Required(FloatArray)
    abs_hiding_end_times        = Required(FloatArray)

    abs_engaged_observ_start_times  = Optional(FloatArray)
    abs_engaged_observ_end_times    = Optional(FloatArray)

    abs_grooming_observ_start_times = Optional(FloatArray)
    abs_grooming_observ_end_times   = Optional(FloatArray)

    abs_resting_observ_start_times  = Optional(FloatArray)
    abs_resting_observ_end_times    = Optional(FloatArray)

    frametimes1 = Optional(FloatArray)
    frametimes2 = Optional(FloatArray)

    x1 = Optional(FloatArray)
    y1 = Optional(FloatArray)
    x2 = Optional(FloatArray)
    y2 = Optional(FloatArray)

    call_times = Optional(FloatArray)
    call_types = Optional(StrArray)

    # animal and date completely define a session
    # enforce uniqueness on them
    composite_key(animal, date)


    @property
    def recorded_cells(self):
        """Cells recorded in the session ordered by their id"""
        return self._recorded_cells.order_by(lambda c: c.id)


    @property
    def all_time_points(self):
        """All tagged time points as np.array"""
        return np.array(self._all_time_points)


    @property
    def trials(self):
        """All trials ordered by time"""
        return self._trials.order_by(lambda t: t.abs_end_time)


    @property
    def seek_trials(self):
        """Seek trials ordered by time"""
        return [t for t in self.trials if t.role == 'seek']


    @property
    def hide_trials(self):
        """Hide trials ordered by time"""
        return [t for t in self.trials if t.role == 'hide']


    @property
    def observing_trials(self):
        """Observing trials ordered by time"""
        return [t for t in self.trials if t.observing]


    @property
    def playing_trials(self):
        """Playing trials ordered by time"""
        return [t for t in self.trials if not t.observing]


    @property
    def last_trial_end(self):
        """End of the last trial"""
        return max([t.abs_end_time for t in self.trials])


def connect_to_db(db_path=None, delete_previous=False):
    """
    Connect to database and generate mapping

    Parameters
    ----------
    db_path : str
        path to the database file
        if None: extract from environment file
    delete_previous : bool, default False
        whether to delete an existing database
        found at db_path

    Returns
    -------
    None
    """
    if db_path is None:
        db_path = os.getenv("DB_PATH")

    if delete_previous:
        if os.path.exists(db_path):
            os.remove(db_path)

        db.bind(provider='sqlite', filename=db_path, create_db=True)
    else:
        db.bind(provider='sqlite', filename=db_path, create_db=False)

    db.generate_mapping(create_tables=True)

