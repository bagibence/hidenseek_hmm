import umap
import xarray as xr

from hidenseek.util.array_utils import split_array

def embed_with_umap(session, n_components=3, n_neighbors=40):
    """
    Embed a session's firing rates with UMAP

    Parameters
    ----------
    session : Session
        session whose firing rates to embed
    n_components : int, default 3
        number of dimensions to embed into
    n_neighbors : int, default 40
        n_neighbors parameter of UMAP

    Returns
    -------
    None, but adds .states and .embeddings to session
    """
    session_factors = xr.concat([trial.factors.drop('time') for trial in session.trials], dim = 'time')

    embedder = umap.UMAP(n_components = n_components, n_neighbors = n_neighbors)
    session_embeddings = embedder.fit_transform(session_factors.transpose('time', 'factor'))

    trial_embeddings = split_array(session_embeddings,
                                   [trial.factors.time.size for trial in session.trials])

    for (trial, emb) in zip(session.trials, trial_embeddings):
        trial.embedding = xr.DataArray(emb, dims = ['time', 'factor'], coords = {'time' : trial.factors.time})
        
    session.states = xr.concat([t.states.drop('time') for t in session.trials], dim = 'time')
    session.embeddings = session_embeddings

