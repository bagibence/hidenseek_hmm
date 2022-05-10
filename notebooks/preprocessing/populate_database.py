# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python (hidenseek)
#     language: python
#     name: hidenseek
# ---

# %% [markdown]
# # Load the data from matlab

# %%
import os

from hidenseek.matlab_interface import *
import hidenseek.preprocessing as preproc
from hidenseek.db_interface import *

# %%
root_dir = os.getenv('ROOT_DIR')
data_dir = os.getenv('DATA_DIR')
interim_data_dir = os.getenv('INTERIM_DATA_DIR')

# %%
cells_data = loadmat(os.path.join(interim_data_dir, 'cells_data_minimal.mat'))['cells_data']
cells = [mCell(d) for d in cells_data]

# %%
sessions_data = loadmat(os.path.join(interim_data_dir, 'sessions_data_minimal.mat'))['sessions_data']
sessions = [mSession(s) for s in sessions_data]

# %% [markdown]
# # Exclude session 6 with 1 recorded neuron

# %%
sessions = [s for s in sessions if s.session_id != 6]

# %% [markdown]
# # Connect to the database and create sessions, cells, trials

# %%
connect_to_db(os.path.join(interim_data_dir, 'database.db'), delete_previous=True)

# %%
add_to_end = 2000

# %%
for (paper_id, s) in enumerate(sessions, start=1):
    
    Session(id                          = s.session_id,
            animal                      = s.animal,
            date                        = s.date,
            paper_id                    = paper_id,
            _all_time_points            = s.all_time_points,
            abs_darting_start_times     = s.abs_darting_start_times,    
            abs_darting_end_times       = s.abs_darting_end_times,      
            abs_interaction_start_times = s.abs_interaction_start_times,
            abs_interaction_end_times   = s.abs_interaction_end_times,  
            abs_jump_in_times           = s.abs_jumpin_times,
            abs_jump_out_times          = s.abs_jumpout_times,            
            abs_box_open_times          = s.abs_box_open_times,          
            abs_box_closed_times        = s.abs_box_closed_times,        
            abs_transit_start_times     = s.abs_transit_start_times,     
            abs_transit_end_times       = s.abs_transit_end_times,      
            abs_sighting_times          = s.abs_sighting_times,         
            abs_exploring_start_times   = s.abs_exploring_start_times,  
            abs_exploring_end_times     = s.abs_exploring_end_times,    
            abs_hiding_start_times      = s.abs_hiding_start_times,     
            abs_hiding_end_times        = s.abs_hiding_end_times,
            frametimes1                 = s.frametimes1,
            frametimes2                 = s.frametimes2,
            x1                          = s.x1,
            y1                          = s.y1,
            x2                          = s.x2,
            y2                          = s.y2,
            call_times                  = s.call_times,
            call_types                  = s.call_types)
    
    assert len(s.seek_trial_start_times) == len(s.seek_trial_end_times)
    for start, end in zip(s.seek_trial_start_times, s.seek_trial_end_times):
        Trial(role = 'seek',
              session = Session.get(id = s.session_id),
              abs_start_time = start,
              abs_end_time = end + add_to_end)
        
    assert len(s.hide_trial_start_times) == len(s.hide_trial_end_times)
    for start, end in zip(s.hide_trial_start_times, s.hide_trial_end_times):
        Trial(role = 'hide',
              session = Session.get(id = s.session_id),
              abs_start_time = start,
              abs_end_time = end + add_to_end)

# %%
for c in cells:
    try:
        Cell(id=c.new_id, 
             cluster_id=c.cluster_id,
             session=Session.get(id=c.session_id),
             _all_spikes=c.all_spikes)
    except:
        print((c.new_id, c.session_id))

# %%
db.commit()

# %% [markdown]
# # Test 
#
# Have to restart the kernel for this.

# %% {"tags": []}
try:
    connect_to_db(os.path.join(interim_data_dir, 'database.db'))
except:
    pass

# %%
for s in Session.select():
    print((s.id, s.date, s.animal, s.recorded_cells.count()))

# %%
