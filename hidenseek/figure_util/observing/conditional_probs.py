import numpy as np


def p_role_in_session(session, role):
    all_time_points = sum([len(trial.states) for trial in session.trials])
    if role == 'observing':
        return sum([len(trial.states) for trial in session.observing_trials]) / all_time_points
    if role == 'playing':
        return sum([len(trial.states) for trial in session.playing_trials]) / all_time_points


def p_state_in_session(session, state_id):
    all_time_points = sum([len(trial.states) for trial in session.trials])
    
    return np.sum([np.sum(trial.states == state_id) for trial in session.trials]) / all_time_points


def p_state_given_role_in_session(session, state_id, role):
    role_trials = getattr(session, f'{role}_trials')
    role_time_points = np.sum([len(trial.states) for trial in role_trials])
    
    return np.sum([np.sum(trial.states == state_id) for trial in role_trials]) / role_time_points


def p_role_given_state_in_session(session, state_id, role):
    return p_state_given_role_in_session(session, state_id, role) * p_role_in_session(session, role) / p_state_in_session(session, state_id)
