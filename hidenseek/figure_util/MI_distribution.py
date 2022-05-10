from hidenseek.db_interface import *

import seaborn as sns
import matplotlib as mpl
import pandas as pd


def count_low_chance_scores(scores_df):
    return pd.Series({session_id : (scores_df.query(f'session_id == {session_id} and kind == "fake"').score >= scores_df.query(f'session_id == {session_id} and kind == "real"').score.values[0]).sum()
            for session_id in scores_df.session_id.unique()})

