import numpy as np
import xarray as xr

from hidenseek.util.postproc.misc import contingency_xr

from hidenseek.globals import (
        behavioral_state_names_seek,
        behavioral_state_names_hide,
        behavioral_state_names_extra,
        behavioral_state_dict,
        inverse_behavioral_state_dict,
        behavioral_state_plot_dict
)


def to_plot_df(cont_mx_xr, rename=False):
    """
    Turn contingency matrix stored as an xarray.DataArray into a
    wide-form pandas.DataFrame that is nice to plot with seaborn.heatmap
    
    Parameters
    ----------
    cont_mx_xr : xr.DataArray
        contingency matrix with dims behavioral_states and hmm_state
    rename : bool, default False
        rename behavioral states to their pretty version with spaces
        
    Returns
    -------
    cont_mx_df : pd.DataFrame
        contingency matrix as a wide dataframe
    """
    cont_mx_xr = cont_mx_xr.assign_coords({'behavioral_state' : [inverse_behavioral_state_dict[i] for i in cont_mx_xr.behavioral_state.values]})
    cont_mx_df = cont_mx_xr.to_dataframe('contingency').reset_index().pivot_table(index = 'behavioral_state', columns = 'hmm_state').contingency
    
    if rename:
        cont_mx_df = cont_mx_df.rename(behavioral_state_plot_dict)
        cont_mx_df = cont_mx_df.rename_axis(columns = {'hmm_state' : 'HMM state'}, index = {'behavioral_state' : 'Behavioral state'})

    return cont_mx_df

from statsmodels.stats.multitest import multipletests

def make_signif_cont_df(p_cont_df, target_p_value, method):
    """
    Make dataframe that shows if a pair is significant or not
    
    Parameters
    ----------
    p_cont_df : pd.DataFrame
        dataframe with the pseudo p-values
    target_p_value : float
        target p-value we aim for
    method : str
        method for correcting for multiple comparisons
        if None, no correction is applied
        
    Returns
    -------
    signif_cont_df : pd.DataFrame
        boolean dataframe that is True
        for significant state pairs
    """
    if method is None:
        signif_cont_df = p_cont_df < target_p_value
    else:
        vals =  (multipletests(p_cont_df.values.flatten(),
                              method = method,
                              alpha = target_p_value)[0]
                 .reshape(-1, p_cont_df.shape[1]))

        signif_cont_df = p_cont_df.copy(deep=True)
        signif_cont_df.loc[:] = vals
    
    return signif_cont_df


def make_cont_dfs(session, trials_to_use, behavior_type, p_value, rename=False, correction=None):
    """
    Make contingency dataframes for session

    Parameters
    ----------
    trials_to_use : str
        observing, playing or both
    behavior_type : str
        observing, playing or both
        if you want to use it on the original dataset, set both to playing
    p_value : float
        p-value used for thresholding the significance matrices
    rename : bool, default False
        rename underscored names to prettier ones with spaces
    correction : str, default None
        method to use for correcting for multiple comparisons
        None means no correction

    Returns
    -------
    cont_df, cont_df_hmm, cont_df_beh, signif_cont_df, signif_cont_df_hmm, signif_cont_df_beh
    """
    if trials_to_use in ['playing', 'both'] and behavior_type == 'observing':
        raise ValueError
    used_trials = [trial for trial in session.trials if trials_to_use in ('both', trial.observing_role)]

    N_vals = {len(trial.fake_states_list) for trial in used_trials} 
    assert len(N_vals) == 1
    N = list(N_vals)[0]
        
    # include trial's states if we're looking at all trials or the trial's type
    hmm_states = np.concatenate([trial.states for trial in used_trials])

    if behavior_type == 'both':
        # the behavior state is playing_states in playing trials and observing_states in observing trials
        behavior_states = np.concatenate([getattr(trial, f"{trial.observing_role}_states") for trial in used_trials])
    else:
        behavior_states = np.concatenate([getattr(trial, f"{behavior_type}_states") for trial in used_trials])

    fake_states_list = [np.concatenate([trial.fake_states_list[i] for trial in used_trials])
                        for i in range(N)]
    
    # contingencies of the real states and behavirs
    real_cont_mx = contingency_xr(behavior_states, hmm_states)
    real_cont_mx_hmm = (real_cont_mx / real_cont_mx.sum('behavioral_state'))
    real_cont_mx_beh = (real_cont_mx / real_cont_mx.sum('hmm_state'))

    # contingencies in every repetition of the fake state generation
    fake_cont_matrices = []
    fake_cont_matrices_hmm = []
    fake_cont_matrices_beh = []
    for fake_states in fake_states_list:
        fake_cont_mx = contingency_xr(behavior_states, fake_states)

        fake_cont_matrices.append(fake_cont_mx)
        fake_cont_matrices_hmm.append(fake_cont_mx / fake_cont_mx.sum('behavioral_state'))
        fake_cont_matrices_beh.append(fake_cont_mx / fake_cont_mx.sum('hmm_state'))

    # combine them into a 3D tensor
    fake_cont_mx = xr.concat(fake_cont_matrices, dim = 'repetition')
    fake_cont_mx_hmm = xr.concat(fake_cont_matrices_hmm, dim = 'repetition')
    fake_cont_mx_beh = xr.concat(fake_cont_matrices_beh, dim = 'repetition')

    # (fake_cont_mx > real_cont_mx).mean('repetition') is the
    # ratio of repetitions with a higher score than the real score
    p_cont_mx = ((fake_cont_mx >= real_cont_mx).mean('repetition'))
    p_cont_mx_hmm = ((fake_cont_mx_hmm >= real_cont_mx_hmm).mean('repetition'))
    p_cont_mx_beh = ((fake_cont_mx_beh >= real_cont_mx_beh).mean('repetition'))

    # turn each array into a nicely plotted dataframe
    cont_df = to_plot_df(real_cont_mx, rename)
    cont_df_hmm = to_plot_df(real_cont_mx_hmm, rename)
    cont_df_beh = to_plot_df(real_cont_mx_beh, rename)

    p_cont_df = to_plot_df(p_cont_mx, rename)
    p_cont_df_hmm = to_plot_df(p_cont_mx_hmm, rename)
    p_cont_df_beh = to_plot_df(p_cont_mx_beh, rename)
    
    # correct for multiple comparisons and make a mask signaling significance
    signif_cont_df = make_signif_cont_df(p_cont_df, p_value, correction)
    signif_cont_df_hmm = make_signif_cont_df(p_cont_df_hmm, p_value, correction)
    signif_cont_df_beh = make_signif_cont_df(p_cont_df_beh, p_value, correction)
    
    return cont_df, cont_df_hmm, cont_df_beh, signif_cont_df, signif_cont_df_hmm, signif_cont_df_beh
