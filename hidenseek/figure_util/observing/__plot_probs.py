import seaborn as sns


def plot_probs_swarm(probs_df, ax_obs_prob_swarm, size=3, remove_xlabel=False):
    sns.swarmplot(y = 'observing', x = 'session_id', data = probs_df, ax = ax_obs_prob_swarm, size = size, color = 'tab:blue')
    ax_obs_prob_swarm.axhline(y = 0.5, linestyle = '--', alpha = 0.5, color = 'grey')
    sns.despine(ax = ax_obs_prob_swarm)
    
    ax_obs_prob_swarm.set_ylabel('P(obs | state)')
    ax_obs_prob_swarm.set_xlabel('session')
    
    if remove_xlabel:
        ax_obs_prob_swarm.set_xlabel('')
        ax_obs_prob_swarm.set_xticks([])
