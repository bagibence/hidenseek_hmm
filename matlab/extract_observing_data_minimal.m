%%
clear all

data_path

raw_data_dir = fullfile(data_dir, 'raw');

mat_paths = {fullfile(raw_data_dir, 'brainplay/M1/20190206/WokspaceSyncNacho.mat'),
             fullfile(raw_data_dir, 'brainplay/M3/20190210/WokspaceSyncNacho.mat'),
             fullfile(raw_data_dir, 'brainplay/M4/20190215/WokspaceSyncNacho.mat'),
             fullfile(raw_data_dir, 'brainplay/M4/20190212/WokspaceSyncNacho.mat'),
             fullfile(raw_data_dir, 'brainplay/M2/20190206/WokspaceSyncNacho.mat'),
             fullfile(raw_data_dir, 'brainplay/M2/20190205/WokspaceSyncNacho.mat'),
             fullfile(raw_data_dir, 'brainplay/M1/20190128/WokspaceSyncNacho.mat'),
             fullfile(raw_data_dir, 'brainplay/M1/20190130/WokspaceSyncNacho.mat'),
             fullfile(raw_data_dir, 'brainplay/M2/20190131/WokspaceSyncNacho.mat'),
             fullfile(raw_data_dir, 'brainplay/M4/20190221/WokspaceSyncNacho.mat'),
             fullfile(raw_data_dir, 'brainplay/M3/20190213/WokspaceSyncNacho.mat'),
             fullfile(raw_data_dir, 'brainplay/M3/20190228/WokspaceSyncNacho.mat')};

%%


counthide=0;
countseek=0;
countcells=0;
for session_ind = 1:length(mat_paths)
    load(mat_paths{session_ind})
    split_path = strsplit(mat_paths{session_ind}, '/');

    sessions_data(session_ind).animal         = split_path(end-2);
    sessions_data(session_ind).date           = split_path(end-1);
    sessions_data(session_ind).session_id     = session_ind;
    sessions_data(session_ind).recorded_cells = goodclusters;
    sessions_data(session_ind).spike_times    = {};

    sessions_data(session_ind).all_time_points         = sort([BoxClosedTimeSync', BoxOpenTimeSync', DartingStartTimeSync', DartingEndTimeSync', InteractionStartTimeSync', InteractionEndTimeSync', JumpInTimeSync', JumpOutTimeSync', SightingStartTimeSync', TransitStartTimeSync', TransitEndTimeSync', ExploringStartTimeSync', ExploringEndTimeSync', HidingEndTimeSync', HidingStartTimeSync']);
    
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

    sessions_data(session_ind).engaged_observ_start_times  = sort(EngagedObservStartTimeSync');
    sessions_data(session_ind).engaged_observ_end_times    = sort(EngagedObservEndTimeSync');
    sessions_data(session_ind).grooming_observ_start_times = sort(GroomingObservStartTimeSync');
    sessions_data(session_ind).grooming_observ_end_times   = sort(GroomingObservEndTimeSync');
    sessions_data(session_ind).resting_observ_start_times  = sort(RestingObservStartTimeSync);
    sessions_data(session_ind).resting_observ_end_times    = sort(RestingObservEndTimeSync');

    sessions_data(session_ind).x1 = RatX1;
    sessions_data(session_ind).y1 = RatY1;

    for Cell = 1:length(goodclusters)
        countcells = countcells + 1;
        clear SpikeTimesInMS
        clustused = goodclusters(Cell);
        SpikeTimesInMS = double(ST(Clusters==clustused)) / 32;

        sessions_data(session_ind).spike_times{end+1} = SpikeTimesInMS;
    end



end

build_observing_cells_data

% save data to disk
interim_data_dir = fullfile(data_dir, 'interim');
if ~exist(interim_data_dir, 'dir')
    mkdir(interim_data_dir)
end
save(fullfile(interim_data_dir, 'observing_sessions_data.mat'), 'sessions_data');

