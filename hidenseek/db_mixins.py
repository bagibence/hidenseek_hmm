import hidenseek.preprocessing as preproc
import hidenseek.globals as globals
from hidenseek.util import postproc

import numpy as np
import pandas as pd
import xarray as xr

import warnings

from functools import cached_property

class TrialMixin:
    """
    Mixin class for adding extra functionality to Trial objects
    """

    def filter(self, array, inclusive=True):
        """
        Filter an array between the start and end of the trial

        Parameters
        ----------
        array : array-like
            array to filter
        inclusive : bool, default True
            whether to include the start and end

        Returns
        -------
        filtered array
        converted to np.array if the original array was list or tuple
        """
        return preproc.filter_between(array, self.abs_start_time, self.abs_end_time, inclusive)


    def __getattr__(self, name):
        try:
            # we need this for properties that end with times or time
            # but need a different behavior
            self.__getattribute__(name)
        except:
            if name.endswith('times') or name.endswith('time'):
                attr_name = name
                if name.endswith('time'):
                    attr_name += 's'
                if not name.startswith('abs_'):
                    attr_name = 'abs_' + attr_name

                abs_times = np.array(getattr(self.session, attr_name))
                trial_abs_times = self.filter(abs_times, inclusive=True)
                
                if not name.startswith('abs_'):
                    return_times = trial_abs_times - self.abs_start_time
                else:
                    return_times = trial_abs_times

                if name.endswith('time'):
                    if len(return_times) > 1:
                        warnings.warn(f"There are more than 1 {name.replace('_', ' ')}s")
                        print(f"There are more than 1 {name.replace('_', ' ')}s")
                    return return_times[0]
                else:
                    return return_times

            # else original behavior
            return self.__getattribute__(name)


    @property
    def all_abs_time_points(self):
        """
        Any kind of tagged time point's absolute time
        that happened inside this trial
        """
        return self.filter(self.session.all_time_points)


    @property
    def all_time_points(self):
        """
        Any kind of tagged time point's time
        that happened inside this trial
        """
        return self.all_abs_time_points - self.abs_start_time


    def get_abs_time_points(self, box_open=True):
        """
        Get the absolute times of time points in the trial that signal the phases of the game

        Parameters
        ----------
        box_open : bool, default True
            whether to include the box open point

        Returns
        -------
        pd.Series with time point names as index
        """
        if self.role == 'seek' and box_open:
            tp_names = globals.time_point_names_bo
        else:
            tp_names = globals.time_point_names
        
        time_points = [self.abs_start_time]
        for tp_name in tp_names[1:-1]:
            try:
                times = self.filter(getattr(self.session, 'abs_' + tp_name + '_times'))
            except:
                times = self.filter(getattr(self.session, 'abs_' + tp_name + '_start_times'))

            if len(times) == 0:
                time_points.append(np.nan)
            else:
                time_points.append(times[0])

        time_points.append(self.abs_end_time)

        return pd.Series(time_points, index = tp_names)


    def get_time_points(self, box_open=True):
        """
        Get the time points in the trial that signal the phases of the game
        relative to trial start

        Parameters
        ----------
        box_open : bool, default True
            whether to include the box open point

        Returns
        -------
        pd.Series with time point names as index
        """
        return self.get_abs_time_points(box_open=box_open) - self.abs_start_time

    @property
    def abs_time_points(self):
        """
        pd.Series with the absolute times of time points that signal the phases of the game
        includes box open point for seek trials
        if you want to exclude them, use get_abs_time_points
        """
        return self.get_abs_time_points(box_open=True)
    
    @property
    def time_points(self):
        """
        pd.Series with the time points that signal the phases of the game
        relative to the start of the trial
        includes box open point for seek trials
        if you want to exclude them, use get_time_points
        """
        return self.abs_time_points - self.abs_start_time


    @property
    def successful(self):
        """
        If there is no tagged interaction point, the trial was failed
        """
        return not np.isnan(self.time_points.interaction)

    
    def get_spike_trains(self, bin_length):
        """
        Bin spikes of all neurons during this trial

        Parameters
        ----------
        bin_length : float
            length of the window used for binning the spikes

        Returns
        -------
        binned spikes as a
        neuron x time xr.DataArray
        """
        return xr.concat([preproc.create_spike_train(st, self.time_points.end, bin_length)
                          for st in self.spike_times],
                         dim = 'neuron')


    def get_smooth_rates(self, bin_length, smoothing_win=None, smoothing_hw=None):
        """
        Get smooth rates of all neurons during the trial

        Parameters
        ----------
        bin_length : float
            length of the window used for binning the spikes
        smoothing_win : array-like, default None
            smoothing window
            e.g. see preproc.norm_gauss_window
            provide either this or smoothing_hw
        smoothing_hw : float, default None
            convolve with a normalized Gaussian window with this half width in ms
            provide either this or smoothing_win

        Returns
        -------
        smoothed rates as a
        neuron x time xr.DataArray
        """
        if (smoothing_hw is not None) and (smoothing_win is not None):
            raise Exception('only provide smoothing_hw or smoothing_win')

        trial_rates = []
        for st in self.spike_times:
            trial_rates.append(preproc.create_smoothed_rate(st, self.time_points.end, bin_length, smoothing_win, smoothing_hw))

        return xr.concat(trial_rates, dim='neuron')


    @cached_property
    def pos1(self):
        """
        Position of the rat on camera #1
        xr.Dataset with time coordinates and variables 'x' and 'y'
        """
        pos = self.session.pos1.sel(time = slice(self.abs_time_points.start,
                                                 self.abs_time_points.end))
        pos['time'] = pos.time - self.abs_time_points.start
        return pos

    @cached_property
    def pos2(self):
        """
        Position of the rat on camera #2
        xr.Dataset with time coordinates and variables 'x' and 'y'
        """
        pos = self.session.pos2.sel(time = slice(self.abs_time_points.start,
                                                  self.abs_time_points.end))
        pos['time'] = pos.time - self.abs_time_points.start
        return pos

    @property
    def calls(self):
        calls_xr = self.session.calls.sel(time = slice(self.abs_time_points.start, self.abs_time_points.end))
        calls_xr['time'] = calls_xr.time - self.abs_time_points.start
        return calls_xr

class SessionMixin:

    @property
    def abs_observing_times(self):
        """Every time point associated with an observing behavior"""
        return np.concatenate([getattr(self, attr) for attr in dir(self) if '_observ_' in attr])

    @property
    def abs_observing_start_times(self):
        """Every time point where an observing behavior starts"""
        return np.concatenate([getattr(self, attr) for attr in dir(self) if ('_observ_' in attr and 'start' in attr)])

    @property
    def abs_observing_end_times(self):
        """Every time point where an observing behavior ends"""
        return np.concatenate([getattr(self, attr) for attr in dir(self) if ('_observ_' in attr and 'end' in attr)])


    def get_time_points(self, role, box_open=True):
        """
        Get time points of every trial with a given role of the session as a DataFrame

        Parameters
        ----------
        role : str
            seek or hide
        box_open : bool, default True
            Include the box open point

        Returns
        -------
        DataFrame with the time points as columns and trials as rows
        """
        if role == 'seek':
            return pd.DataFrame([t.get_time_points(box_open) for t in self.seek_trials])
        elif role == 'hide':
            return pd.DataFrame([t.time_points for t in self.hide_trials])
        elif role == 'both':
            return pd.DataFrame([t.get_time_points(box_open) for t in self.trials])
        else:
            raise Exception('role has to be either seek or hide or both')


    def get_median_time_points(self, role, box_open=True, dropna=True):
        """
        Median of the time points across trials of a given role
        in the session 

        Parameters
        ----------
        role : str
            seek or hide
        box_open : bool, default True
            whether to include the box open point in seek
        dropna : bool, default True
            disregard trials where some time points are NaN

        Returns
        -------
        pd.Series with the median time points
        """
        if dropna:
            return self.get_time_points(role, box_open).dropna().median()
        else:
            return self.get_time_points(role, box_open).median(skipna = True)


    def get_spike_trains(self, bin_length):
        """
        Extract spike trains by binning the spiketimes of every cell in the session

        Parameters
        ----------
        bin_length : int
            length of the bins in ms

        Returns
        -------
        neuron x time xr.DataArray
        """
        return xr.concat([preproc.create_spike_train(c.all_spikes, self.last_trial_end+bin_length, bin_length)
                         for c in self.recorded_cells],
                         dim='neuron')


    def get_stretched_states(self, bin_length, median_time_points_seek=None, median_time_points_hide=None, orig=False):
        """
        Stretch most likely states of every trial of a given session to the same length
        Assumes states have been added to the trials
        
        Parameters
        ----------
        bin_length : float
            spike binning length
        median_time_points_seek : pd.Series, optional
            median time points in the seek trials
        median_time_points_hide : pd.Series, optional
            median time points in the hide trials
        orig : bool, default False
            use the original states saved in orig_states instead of states
            useful is we want to plot the states using their original coloring
            but have already permuted them to match a reference session
        
        Returns
        -------
        stretched_states_seek : xr.DataArray
            trial x time DataArray containing the states in the seek trials stretched to the same length
        stretched_states_hide : xr.DataArray
            trial x time DataArray containing the states in the hide trials stretched to the same length
        """
        if median_time_points_seek is None:
            median_time_points_seek = self.get_median_time_points('seek', True)
        if median_time_points_hide is None:
            median_time_points_hide = self.get_median_time_points('hide', True)

        box_open = ('box_open' in median_time_points_seek.index)

        stretched_states_seek = []
        for trial in self.seek_trials:
            trial_states = trial.orig_states if orig else trial.states
            states = postproc.stretch_states(trial_states, trial.get_time_points(box_open).values, median_time_points_seek.values, bin_length)
            stretched_states_seek.append(states)
            
        stretched_states_hide = []
        for trial in self.hide_trials:
            trial_states = trial.orig_states if orig else trial.states
            states = postproc.stretch_states(trial_states, trial.time_points.values, median_time_points_hide.values, bin_length)
            stretched_states_hide.append(states)
            
        stretched_states_seek = xr.concat(stretched_states_seek, dim='trial')
        stretched_states_hide = xr.concat(stretched_states_hide, dim='trial')

        stretched_states_seek['trial'] = [t.id for t in self.seek_trials]
        stretched_states_hide['trial'] = [t.id for t in self.hide_trials]
        
        return stretched_states_seek, stretched_states_hide


    def get_state_probs(self, bin_length, median_time_points_seek=None, median_time_points_hide=None):
        """
        Align and stretch the state probabilities of the individual trials to the same length and concatenate 
        them into a DataArray both for seek and hide
        Assumes state probabilities have been added to trials

        Parameters
        ----------
        bin_length : float
            spike binning length
        median_time_points_seek : pd.Series, optional
            median time points in the seek trials
        median_time_points_hide : pd.Series, optional
            median time points in the hide trials
        
        Returns
        -------
        stretched_states_seek : xr.DataArray
            trial x time DataArray containing the states in the seek trials stretched to the same length
        stretched_states_hide : xr.DataArray
            trial x time DataArray containing the states in the hide trials stretched to the same length
        """
        if median_time_points_seek is None:
            median_time_points_seek = self.get_median_time_points('seek', True)
        if median_time_points_hide is None:
            median_time_points_hide = self.get_median_time_points('hide', True)

        box_open = ('box_open' in median_time_points_seek.index)

        stretched_state_probs_seek = [postproc.stretch_2d(trial.state_probs, trial.get_time_points(box_open).values, median_time_points_seek, bin_length)
                                      for trial in self.seek_trials]

        stretched_state_probs_hide = [postproc.stretch_2d(trial.state_probs, trial.time_points.values, median_time_points_hide, bin_length)
                                      for trial in self.hide_trials]

        state_probs_seek = xr.concat(stretched_state_probs_seek, dim = 'trial')
        state_probs_hide = xr.concat(stretched_state_probs_hide, dim = 'trial')

        return state_probs_seek, state_probs_hide 

    
    def get_state_hist(self, bin_length, K, median_time_points_seek_bo=None, median_time_points_hide=None):
        """
        Calculate probabilities of the states occurring in time with the trials of the session stretched to the same length
        and aligned on the median time points in the session
        Assumes the states have been added to the trials

        Parameters
        ----------
        bin_length : float
            spike binning length
        K : int
            number of states
        median_time_points_seek_bo : pd.Series, optional
            median time points in seek with box open
        median_time_points_hide : pd.Series, optional
            median time points in hide

        Returns
        -------
        state_hist_seek : xr.DataArray
            state x time array
            number of times the state was active at a time after stretching the trials to the same length
        """
        if median_time_points_seek_bo is None:
            median_time_points_seek_bo = self.get_median_time_points('seek', True)
        if median_time_points_hide is None:
            median_time_points_hide = self.get_median_time_points('hide', True)

        stretched_states_seek, stretched_states_hide = self.get_stretched_states(bin_length,
                                                                                 median_time_points_seek_bo,
                                                                                 median_time_points_hide)

        state_hist_seek = postproc.get_state_hist(stretched_states_seek, K, True)
        state_hist_hide = postproc.get_state_hist(stretched_states_hide, K, True)

        return state_hist_seek, state_hist_hide


    @property
    def observing_trial_ids(self):
        """
        IDs of the observing trials in the session
        """
        return sorted([t.id for t in self.trials if t.observing])


    @property
    def playing_trial_ids(self):
        """
        IDs of the playing trials in the session
        """
        return sorted([t.id for t in self.trials if not t.observing])

    @cached_property
    def pos1(self):
        """
        Position of the rat on camera #1
        xr.Dataset with time coordinates and variables 'x' and 'y'
        """
        return xr.Dataset({'x' : xr.DataArray(self.x1, dims = ['time'], coords = {'time' : self.frametimes1}),
                           'y' : xr.DataArray(self.y1, dims = ['time'], coords = {'time' : self.frametimes1})})

    @cached_property
    def pos2(self):
        """
        Position of the rat on camera #2
        xr.Dataset with time coordinates and variables 'x' and 'y'
        """
        return xr.Dataset({'x' : xr.DataArray(self.x2, dims = ['time'], coords = {'time' : self.frametimes2}),
                           'y' : xr.DataArray(self.y2, dims = ['time'], coords = {'time' : self.frametimes2})})

    @cached_property
    def calls(self):
        return xr.DataArray(self.call_types, dims = ['time'], coords = {'time' : self.call_times})
