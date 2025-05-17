% Load all 4 groups
fc = load('FC.mat');        % Female controls
mc = load('MC.mat');        % Male controls
fa = load('FADHD.mat');     % Female ADHD
ma = load('MADHD.mat');     % Male ADHD


% Eyes Open Baseline
fc_eo1 = FC{1};    % Female controls
fc_eo2 = FC{7};
mc_eo1 = MC{1};
mc_eo2 = MC{7};
fa_eo1 = FADHD{1};
fa_eo2 = FADHD{7};
ma_eo1 = MADHD{1};
ma_eo2 = MADHD{7};

% Eyes Closed Baseline
fc_ec1 = FC{2};
fc_ec2 = FC{8};
mc_ec1 = MC{2};
mc_ec2 = MC{8};
fa_ec1 = FADHD{2};
fa_ec2 = FADHD{8};
ma_ec1 = MADHD{2};
ma_ec2 = MADHD{8};

FADHD{1}(7,:,:) = [];  % EO (task 1)
FADHD{2}(7,:,:) = [];  % EC (task 2)
FADHD{7}(7,:,:) = [];  % EO (task 7)
FADHD{8}(7,:,:) = [];  % EC (task 8)

% Pre-processing pipeline mimicing Automagic
% === Setup ===
fs = 256;  % sampling rate
outputDir = 'preprocessed_subjects';
mkdir(outputDir);

groups = {'FC','MC','FADHD','MADHD'};
task_ids = [1 2 7 8];  % EO1, EC1, EO2, EC2
task_labels = {'EO1','EC1','EO2','EC2'};

% Load data
fc = load('FC.mat'); FC = fc.FC;
mc = load('MC.mat'); MC = mc.MC;
fa = load('FADHD.mat'); FADHD = fa.FADHD;
ma = load('MADHD.mat'); MADHD = ma.MADHD;

% Remove corrupted subject 7 from FADHD tasks
for t = [1 2 7 8]
    FADHD{t}(7,:,:) = [];
end

data_structs = {FC, MC, FADHD, MADHD};

% === Loop and save preprocessed files ===
for g = 1:length(groups)
    group_name = groups{g};
    group_data = data_structs{g};

    for t = 1:length(task_ids)
        task_idx = task_ids(t);
        label = task_labels{t};

        task_data = group_data{task_idx};

        for s = 1:size(task_data,1)
            raw = squeeze(task_data(s,:,:));
            clean = preprocess_eeg(raw, fs);

            % Print progress
            fprintf('Saving %s subject %d task %s\n', group_name, s, label)

            % Save file
            filename = sprintf('%s_subj%02d_task%s.mat', group_name, s, label);
            filepath = fullfile(outputDir, filename);
            save(filepath, 'clean');
        end
    end
end

disp('Preprocessing complete! Check preprocessed_subjects folder.')

% === Preprocessing Function ===
function out = preprocess_eeg(data, fs)
    if ndims(data) == 3
        data = squeeze(data);  % [samples x channels]
    end

    % Bandpass range
    lowcut = 1;
    highcut = 40;

    % FFT-based filtering
    n_samples = size(data,1);
    n_channels = size(data,2);

    % Create frequency vector
    freqs = (0:n_samples-1)*(fs/n_samples);
    freq_mask = (freqs >= lowcut & freqs <= highcut) | ...
                (freqs >= fs - highcut & freqs <= fs - lowcut);

    for ch = 1:n_channels
        sig = data(:,ch);
        fft_sig = fft(sig);
        fft_sig(~freq_mask') = 0;
        filtered = real(ifft(fft_sig));
        data(:,ch) = filtered;
    end

    % Re-reference to average
    data = data - mean(data, 2);

    out = data;
end

% Merging and saving to csv
% === SETTINGS ===
input_folder = 'preprocessed_subjects';
output_file = 'eeg_features.csv';
fs = 256;  % sampling rate in Hz

% Define EEG frequency bands
band_definitions = {
    'Delta',   [1.0 4.0];
    'Theta',   [4.0 8.0];
    'Alpha',   [8.0 12.0];
    'Alpha1',  [8.0 10.0];
    'Alpha2',  [10.0 12.0];
    'Beta',    [12.0 25.0];
    'Beta1',   [12.0 15.0];
    'Beta2',   [15.0 18.0];
    'Beta3',   [18.0 25.0];
    'HighBeta',[25.0 30.0];
    'Gamma',   [30.0 40.0];
    'Gamma1',  [30.0 35.0];
    'Gamma2',  [35.0 40.0];
};

% Get all .mat files from folder
files = dir(fullfile(input_folder, '*.mat'));
output = {};

% === LOOP THROUGH FILES ===
for i = 1:length(files)
    fname = files(i).name;
    load(fullfile(input_folder, fname));  % loads variable 'clean'

    % Parse metadata: group, subject, task
    tokens = regexp(fname, '(\w+)_subj(\d+)_task([A-Z0-9]+)', 'tokens');
    if isempty(tokens)
        warning("Filename not matched: %s", fname);
        continue
    end
    tokens = tokens{1};
    group = tokens{1};
    subject = ['subj', tokens{2}];
    task = tokens{3};

    % Process each channel
    ch_count = size(clean,2);
    for ch = 1:ch_count
        x = double(clean(:,ch));
        N = length(x);

        % === FFT-based Power Spectrum
        X = fft(x);
        pxx = (1/(fs*N)) * abs(X).^2;
        pxx = pxx(1:floor(N/2));
        f = (0:N-1)*(fs/N);
        f = f(1:floor(N/2));

        % === Compute Band Powers
        band_powers = zeros(1, size(band_definitions, 1));
        for b = 1:size(band_definitions, 1)
            range = band_definitions{b,2};
            idx = f >= range(1) & f <= range(2);
            band_powers(b) = trapz(f(idx), pxx(idx));
        end

        % === Compute TBR
        theta = band_powers(strcmp(band_definitions(:,1), 'Theta'));
        beta  = band_powers(strcmp(band_definitions(:,1), 'Beta'));
        TBR = theta / beta;

        % === Store in output table
        channel_label = sprintf('Ch%d', ch);
        output(end+1,:) = [{subject, group, task, channel_label}, num2cell(band_powers), TBR];
    end
end

% === HEADER ROW
band_names = band_definitions(:,1)';
header = [{'Subject', 'Group', 'Task', 'Channel'}, band_names, {'TBR'}];

% === SAVE TO CSV
cell2csv(output_file, [header; output]);
disp(['Features saved to ', output_file]);


% === HELPER FUNCTION ===
function cell2csv(filename, cellArray)
    fid = fopen(filename, 'w');
    for row = 1:size(cellArray,1)
        for col = 1:size(cellArray,2)
            val = cellArray{row,col};
            if isnumeric(val)
                fprintf(fid, '%.6f', val);
            else
                fprintf(fid, '%s', val);
            end
            if col ~= size(cellArray,2)
                fprintf(fid, ',');
            end
        end
        fprintf(fid, '\n');
    end
    fclose(fid);
end



% again
% === SETTINGS ===
input_folder = 'preprocessed_subjects';
output_file = 'eeg_features.csv';
fs = 256;  % sampling rate in Hz

% Define EEG frequency bands
band_definitions = {
    'Delta',   [1.0 4.0];
    'Theta',   [4.0 8.0];
    'Alpha',   [8.0 12.0];
    'Alpha1',  [8.0 10.0];
    'Alpha2',  [10.0 12.0];
    'Beta',    [12.0 25.0];
    'Beta1',   [12.0 15.0];
    'Beta2',   [15.0 18.0];
    'Beta3',   [18.0 25.0];
    'HighBeta',[25.0 30.0];
    'Gamma',   [30.0 40.0];
    'Gamma1',  [30.0 35.0];
    'Gamma2',  [35.0 40.0];
};

% Get all .mat files from folder
files = dir(fullfile(input_folder, '*.mat'));
output = {};

% === LOOP THROUGH FILES ===
for i = 1:length(files)
    fname = files(i).name;
    load(fullfile(input_folder, fname));  % loads variable 'clean'

    % Parse metadata: group, subject, task
    tokens = regexp(fname, '(\w+)_subj(\d+)_task([A-Z0-9]+)', 'tokens');
    if isempty(tokens)
        warning("Filename not matched: %s", fname);
        continue
    end
    tokens = tokens{1};
    group = tokens{1};
    subject = ['subj', tokens{2}];
    task = tokens{3};

    % Process each channel
    ch_count = size(clean,2);
    for ch = 1:ch_count
        x = double(clean(:,ch));
        N = length(x);

        % === FFT-based Power Spectrum
        X = fft(x);
        pxx = (1/(fs*N)) * abs(X).^2;
        pxx = pxx(1:floor(N/2));
        f = (0:N-1)*(fs/N);
        f = f(1:floor(N/2));

        % === Compute Band Powers
        band_powers = zeros(1, size(band_definitions, 1));
        for b = 1:size(band_definitions, 1)
            range = band_definitions{b,2};
            idx = f >= range(1) & f <= range(2);
            band_powers(b) = trapz(f(idx), pxx(idx));
        end

        % === Compute TBR
        theta = band_powers(strcmp(band_definitions(:,1), 'Theta'));
        beta  = band_powers(strcmp(band_definitions(:,1), 'Beta'));
        TBR = theta / beta;

        % === Store in output table
        % Rename only for EO and EC resting tasks
if ismember(task_idx, [1, 2])  % Cz, F4
    if ch == 1
        channel_label = 'Cz';
    else
        channel_label = 'F4';
    end
elseif ismember(task_idx, [7, 8])  % O1, F4
    if ch == 1
        channel_label = 'O1';
    else
        channel_label = 'F4';
    end
else
    continue  % skip all other tasks
end
;
        output(end+1,:) = [{subject, group, task, channel_label}, num2cell(band_powers), TBR];
    end
end

% === HEADER ROW
band_names = band_definitions(:,1)';
header = [{'Subject', 'Group', 'Task', 'Channel'}, band_names, {'TBR'}];

% === SAVE TO CSV
cell2csv(output_file, [header; output]);
disp(['Features saved to ', output_file]);


% AGAIN
% === SETTINGS ===
input_folder = 'preprocessed_subjects';
output_file = 'eeg_features.csv';
fs = 256;  % sampling rate in Hz

% Define EEG frequency bands
band_definitions = {
    'Delta',   [1.0 4.0];
    'Theta',   [4.0 8.0];
    'Alpha',   [8.0 12.0];
    'Alpha1',  [8.0 10.0];
    'Alpha2',  [10.0 12.0];
    'Beta',    [12.0 25.0];
    'Beta1',   [12.0 15.0];
    'Beta2',   [15.0 18.0];
    'Beta3',   [18.0 25.0];
    'HighBeta',[25.0 30.0];
    'Gamma',   [30.0 40.0];
    'Gamma1',  [30.0 35.0];
    'Gamma2',  [35.0 40.0];
};

% Get all .mat files from folder
files = dir(fullfile(input_folder, '*.mat'));
output = {};

segment_len = 2 * fs;    % 2 seconds
step_size = fs;          % 50% overlap

% === LOOP THROUGH FILES ===
for i = 1:length(files)
    fname = files(i).name;
    load(fullfile(input_folder, fname));  % loads variable 'clean'

    % Parse metadata: group, subject, task
    tokens = regexp(fname, '(\w+)_subj(\d+)_task([A-Z0-9]+)', 'tokens');
    if isempty(tokens)
        warning("Filename not matched: %s", fname);
        continue
    end
    tokens = tokens{1};
    group = tokens{1};
    subject = ['subj', tokens{2}];
    task = tokens{3};

    ch_count = size(clean,2);
    N = size(clean,1);

    for ch = 1:ch_count
        x = double(clean(:,ch));

        % Sliding window segmentation
        for start_idx = 1:step_size:(N - segment_len + 1)
            segment = x(start_idx:start_idx + segment_len - 1);
            segment_id = sprintf('%s_seg%d', task, start_idx);

            % FFT-based Power Spectrum
            X = fft(segment);
            pxx = (1/(fs * segment_len)) * abs(X).^2;
            pxx = pxx(1:floor(segment_len/2));
            f = (0:segment_len - 1) * (fs / segment_len);
            f = f(1:floor(segment_len/2));

            % Compute Band Powers
            band_powers = zeros(1, size(band_definitions, 1));
            for b = 1:size(band_definitions, 1)
                range = band_definitions{b,2};
                idx = f >= range(1) & f <= range(2);
                band_powers(b) = trapz(f(idx), pxx(idx));
            end

            % Compute TBR
            theta = band_powers(strcmp(band_definitions(:,1), 'Theta'));
            beta  = band_powers(strcmp(band_definitions(:,1), 'Beta'));
            TBR = theta / beta;

            % Rename only for EO and EC resting tasks
            if ismember(str2double(task(end)), [1, 2])  % Cz, F4
                if ch == 1
                    channel_label = 'Cz';
                else
                    channel_label = 'F4';
                end
            elseif ismember(str2double(task(end)), [7, 8])  % O1, F4
                if ch == 1
                    channel_label = 'O1';
                else
                    channel_label = 'F4';
                end
            else
                continue
            end

            % Store result
            output(end+1,:) = [{subject, group, segment_id, channel_label}, num2cell(band_powers), TBR];
        end
    end
end

% === HEADER ROW
band_names = band_definitions(:,1)';
header = [{'Subject', 'Group', 'SegmentID', 'Channel'}, band_names, {'TBR'}];

% === SAVE TO CSV
cell2csv(output_file, [header; output]);
disp(['Features saved to ', output_file]);
