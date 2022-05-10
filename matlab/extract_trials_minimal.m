%% load data
clear all

data_path

%raw_data_dir = [data_dir 'raw/'];
raw_data_dir = fullfile(data_dir, 'raw');
mat_paths = ...
   {fullfile(raw_data_dir, 'AR01/20180811/WokspaceSyncNacho.mat'), ...
    fullfile(raw_data_dir, 'AR01/20180805/WokspaceSyncNacho.mat'), ...
    fullfile(raw_data_dir, 'AR01/20180807/WokspaceSyncNacho.mat'), ...
    fullfile(raw_data_dir, 'AR06/20180827/WokspaceSyncNacho.mat'), ...
    fullfile(raw_data_dir, 'AR06/20180903/WokspaceSyncNacho.mat'), ...
    fullfile(raw_data_dir, 'AR06/20180825/WokspaceSyncNacho.mat'), ...
    fullfile(raw_data_dir, 'AR06/20180831/WokspaceSyncNacho.mat'), ...
    fullfile(raw_data_dir, 'AR05/20180825/WokspaceSyncNacho.mat'), ...
    fullfile(raw_data_dir, 'AR05/20180824/WokspaceSyncNacho.mat'), ...
    fullfile(raw_data_dir, 'AR05/20180831/WokspaceSyncNacho.mat'), ...
    fullfile(raw_data_dir, 'AR02/20180724/WokspaceSyncNacho.mat'), ...
    fullfile(raw_data_dir, 'AR02/20180711/WokspaceSyncNacho.mat'), ...
    fullfile(raw_data_dir, 'AR02/20180717/WokspaceSyncNacho.mat'), ...
    fullfile(raw_data_dir, 'AR02/20180726/WokspaceSyncNacho.mat')};

% define basic stuff
% could be calculated
N = 177; % number of neurons


for session_ind = 1:length(mat_paths)
%for session_ind = 1:1                 % used for testing on the first session
    % load session data
    load(mat_paths{session_ind});

    
    split_path = strsplit(mat_paths{session_ind}, '/');
    sessions_data(session_ind).animal         = split_path(end-2);
    sessions_data(session_ind).date           = split_path(end-1);
    sessions_data(session_ind).session_id     = session_ind;
    sessions_data(session_ind).recorded_cells = goodclusters;
    sessions_data(session_ind).spike_times    = {};

    sessions_data(session_ind).all_time_points         = sort([SeekTrialStartTimeSync', SeekTrialEndTimeSync', HidingTrialStartTimeSync', HidingTrialEndTimeSync', BoxClosedTimeSync', BoxOpenTimeSync', DartingStartTimeSync', DartingEndTimeSync', InteractionStartTimeSync', InteractionEndTimeSync', JumpInTimeSync', JumpOutTimeSync', SightingStartTimeSync', TransitStartTimeSync', TransitEndTimeSync', ExploringStartTimeSync', ExploringEndTimeSync', HidingEndTimeSync', HidingStartTimeSync']);

    sessions_data(session_ind).seek_trial_start_times     = sort(SeekTrialStartTimeSync');
    sessions_data(session_ind).seek_trial_end_times       = sort(SeekTrialEndTimeSync');

    sessions_data(session_ind).hide_trial_start_times     = sort(HidingTrialStartTimeSync');
    sessions_data(session_ind).hide_trial_end_times       = sort(HidingTrialEndTimeSync');

    sessions_data(session_ind).darting_start_times     = sort(DartingStartTimeSync');
    sessions_data(session_ind).darting_end_times       = sort(DartingEndTimeSync');

    sessions_data(session_ind).interaction_start_times = sort(InteractionStartTimeSync');
    sessions_data(session_ind).interaction_end_times   = sort(InteractionEndTimeSync');

    sessions_data(session_ind).jumpin_times            = sort(JumpInTimeSync');
    sessions_data(session_ind).jumpout_times           = sort(JumpOutTimeSync');

    sessions_data(session_ind).box_open_times          = sort(BoxOpenTimeSync');
    sessions_data(session_ind).box_closed_times        = sort(BoxClosedTimeSync');

    sessions_data(session_ind).transit_start_times     = sort(TransitStartTimeSync');
    sessions_data(session_ind).transit_end_times       = sort(TransitEndTimeSync');

    sessions_data(session_ind).sighting_times          = sort(SightingStartTimeSync');

    sessions_data(session_ind).exploring_start_times   = sort(ExploringStartTimeSync');
    sessions_data(session_ind).exploring_end_times     = sort(ExploringEndTimeSync');

    sessions_data(session_ind).hiding_start_times      = sort(HidingStartTimeSync');
    sessions_data(session_ind).hiding_end_times        = sort(HidingEndTimeSync');

    sessions_data(session_ind).frametimes1 = FrameTimeinLoggerTimeMSCam1;
    sessions_data(session_ind).frametimes2 = FrameTimeinLoggerTimeMSCam2;

    sessions_data(session_ind).x1 = RatX1;
    sessions_data(session_ind).y1 = RatY1;
    sessions_data(session_ind).x2 = RatX2;
    sessions_data(session_ind).y2 = RatY2;

    sessions_data(session_ind).call_times = CallTimeinLoggTime;
    sessions_data(session_ind).call_types = CallType;

    % go through the cells recorded in that session
    % goodclusters is a list of cell IDs unique to that session
    for Cell = 1:length(goodclusters)
        clear SpikeTimesInMS
        clustused = goodclusters(Cell);

        % get spike times of that neuron
        SpikeTimesInMS = double(ST(Clusters == clustused)) / 32;

        sessions_data(session_ind).spike_times{end+1} = SpikeTimesInMS;
    end
end

% build a data structure that is based around cells, not sessions
build_cells_data;

% save data to disk
interim_data_dir = fullfile(data_dir, 'interim');
if ~exist(interim_data_dir, 'dir')
    mkdir(interim_data_dir)
end
save(fullfile(interim_data_dir, 'sessions_data_minimal.mat'), 'sessions_data');

