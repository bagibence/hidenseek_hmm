import numpy as np

def get_p(sdf):
    """
    Calculate pseudo-p-value

    Parameters
    ----------
    sdf : pd.DataFrame
        dataframe with real (just one entry) and fake scores

    Returns
    -------
    ratio of fake scores >= real score
    """
    real_score = sdf.query('kind == "real"').iloc[0].score.item()
    fake_scores = sdf.query('kind == "fake"').score.values

    return np.mean(fake_scores >= real_score)

