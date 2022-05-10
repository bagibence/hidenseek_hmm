%%
% stores the cell IDs per session
% 31 is the max number of recorded neurons
cells_per_session = nan(length(sessions_data), 31);

% count the number of cells
cell_cnt = 0;
% going through sessions
for s_ind = 1:length(sessions_data)
    session = sessions_data(s_ind);
    sessions_data(s_ind).recorded_cells_id = [];
    
    % save the recorded neurons in the array
    cells_per_session(s_ind, 1:length(session.recorded_cells)) = session.recorded_cells;
    
    % go through the cells recorded in this session
    for c_ind = 1:length(session.recorded_cells)
        cell_cnt = cell_cnt + 1;
        cluster_id = session.recorded_cells(c_ind);
        
        % save stuff
        cells_data(cell_cnt).new_id     = cell_cnt;
        cells_data(cell_cnt).cluster_id = cluster_id;
        cells_data(cell_cnt).animal     = session.animal;
        cells_data(cell_cnt).date       = session.date;
        cells_data(cell_cnt).session_id = session.session_id;
        cells_data(cell_cnt).all_spikes = session.spike_times{c_ind};
        
        sessions_data(s_ind).recorded_cells_id(end+1) = cell_cnt;
    end
end

data_path

interim_data_dir = fullfile(data_dir, 'interim');
if ~exist(interim_data_dir, 'dir')
    mkdir(interim_data_dir)
end

disp(['there are ', num2str(cell_cnt), ' cells recorded'])
save(fullfile(interim_data_dir, 'cells_data_minimal.mat'), 'cells_data')

