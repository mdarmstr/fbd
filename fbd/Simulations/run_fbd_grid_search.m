%% FBD reconstruction: every combination of the four grids
clear; clc;

%% Paths
project_dir = fileparts(mfilename('fullpath')); % fbd/Simulations
fbd_dir = fileparts(project_dir);              % fbd

addpath(fbd_dir);
addpath(genpath(fullfile(fbd_dir,'..','MEDA')));

outdir = fullfile(fbd_dir,'results_fbd');
if ~exist(outdir,'dir'), mkdir(outdir); end

vars_pos = 400;
n_perms = 2; % Stored by FBD; no independence test is called here.
n_trials = 5;
base_seed = 1000;
max_attempts=10;

reps_grid = [10,25,50,75,100];
reldim_grid = [0.3,0.4,0.5,0.6,0.7];
levels_grid = [3,4,5,6,7];
nse_grid = [0.1,1,3,6]; % Effect amplitude; residual noise SD remains 1.

trials = table('Size',[0,8], ...
    'VariableTypes',[repmat({'double'},1,7),{'string'}], ...
    'VariableNames',{'Replicates','Split','Levels','Amplitude', ...
    'Trial','Recon','RV','Status'});

results = array2table(zeros(0,10),'VariableNames', ...
    {'Replicates','Split','Levels','Amplitude','ValidRecon','ValidRV', ...
    'ReconMean','ReconSD','RVMean','RVSD'});

%% Full pair search
for reps_pos = reps_grid
    for splitFrac = reldim_grid
        for n_levels = levels_grid
            for nse = nse_grid

                recon_vals = nan(n_trials,1);
                rv_vals = nan(n_trials,1);

                for trial = 1:n_trials
                    recon = NaN;
                    rv = NaN;

                    for attempt = 1:max_attempts
                        % Attempt 1 preserves the original trial seed.
                        % Each retry uses a new dataset.
                        seed = base_seed + trial + (attempt-1)*n_trials;
                        rng(seed,'twister');

                        try
                            [X11,X22,X12,F1,F2] = simulate_blocks(n_levels,reps_pos,vars_pos,nse,splitFrac);

                            [recon,rv,status] = evaluate_reconstruction(X11,X22,X12,F1,F2,n_perms);

                            % The existing helper sometimes returns a failure status
                            % instead of throwing an error. Treat both the same way.
                            if status ~= "ok"
                                error('FBD:InvalidTrial','%s',char(status));
                            end

                            if ~isfinite(recon) || ~isfinite(rv)
                                error('FBD:InvalidMetric','Nonfinite reconstruction or RV.');
                            end

                            status = "ok; attempt " + string(attempt);
                            break;

                        catch ME
                            recon = NaN;
                            rv = NaN;
                            status = "failed: " + string(ME.message);

                            if attempt == max_attempts
                                warning('FBD:RetriesExhausted', ...
                                    'Trial %d failed after %d attempts: %s', ...
                                    trial,max_attempts,ME.message);
                            end
                        end
                    end

                    recon_vals(trial) = recon;
                    rv_vals(trial) = rv;

                    trials(end+1,:) = {reps_pos,splitFrac,n_levels,nse,trial,recon,rv,status}; %#ok
                end

                results(end+1,:) = {reps_pos,splitFrac, n_levels, nse, sum(isfinite(recon_vals)),sum(isfinite(rv_vals)), mean(recon_vals,'omitnan'),std(recon_vals,0,'omitnan'), mean(rv_vals,'omitnan'),std(rv_vals,0,'omitnan')}; %#ok

                fprintf('reps=%d, split=%.1f, levels=%d, amplitude=%g: %d/%d valid reconstructions\n', reps_pos,splitFrac,n_levels,nse, sum(isfinite(recon_vals)),n_trials);
            end
        end
        % Save progress after each replicate/split combination.
        save(fullfile(outdir,'grid_search.mat'),'results','trials','reps_grid','reldim_grid','levels_grid','nse_grid','vars_pos','n_perms','n_trials','base_seed');
    end
end

%% Export
writetable(trials,fullfile(outdir,'grid_trials.csv'));
writetable(results,fullfile(outdir,'grid_summary.csv'));
write_pairwise_tables(results,outdir);
disp(results);

function write_pairwise_tables(results,outdir) %Generated with AI assistance
parameters = {'Replicates','Split','Levels','Amplitude'};
pairs = nchoosek(1:4,2);

pairdir = fullfile(outdir,'pairwise');
if ~exist(pairdir,'dir'), mkdir(pairdir); end

for pair = 1:size(pairs,1)
    row_name = parameters{pairs(pair,1)};
    col_name = parameters{pairs(pair,2)};

    fixed_indices = setdiff(1:4,pairs(pair,:));
    fixed_names = parameters(fixed_indices);

    % One table for each setting of the remaining two parameters.
    fixed_settings = unique(results{:,fixed_names},'rows');

    for setting = 1:size(fixed_settings,1)
        fixed1 = fixed_settings(setting,1);
        fixed2 = fixed_settings(setting,2);

        selected = ...
            results.(fixed_names{1}) == fixed1 & ...
            results.(fixed_names{2}) == fixed2;

        subset = results(selected,:);

        row_values = unique(subset.(row_name));
        col_values = unique(subset.(col_name));

        recon_text = strings(numel(row_values),numel(col_values));
        rv_text = strings(size(recon_text));

        for i = 1:numel(row_values)
            for j = 1:numel(col_values)
                selected_cell = ...
                    subset.(row_name) == row_values(i) & ...
                    subset.(col_name) == col_values(j);

                entry = subset(selected_cell,:);

                recon_text(i,j) = sprintf('%.3f +/- %.3f', ...
                    entry.ReconMean,entry.ReconSD);

                rv_text(i,j) = sprintf('%.3f +/- %.3f', ...
                    entry.RVMean,entry.RVSD);
            end
        end

        filename = sprintf('%s_vs_%s__%s_%g__%s_%g', ...
            row_name,col_name, ...
            fixed_names{1},fixed1,fixed_names{2},fixed2);

        caption = sprintf( ...
            '%s versus %s; %s = %g, %s = %g. Reconstruction error above; RV below.', ...
            row_name,col_name, ...
            fixed_names{1},fixed1,fixed_names{2},fixed2);

        % CSV: both metrics in each cell.
        cell_text = recon_text + " (" + rv_text + ")";
        headers = matlab.lang.makeValidName( ...
            compose('%s_%g',col_name,col_values));

        T = array2table(cell_text,'VariableNames',cellstr(headers));
        T = addvars(T,row_values,'Before',1, ...
            'NewVariableNames',row_name);

        writetable(T,fullfile(pairdir,[filename,'.csv']));

        % LaTeX: two lines per cell, as in the earlier tables.
        write_pairwise_latex( ...
            fullfile(pairdir,[filename,'.tex']),caption, ...
            row_name,col_name,row_values,col_values,recon_text,rv_text);
    end
end
end


function write_pairwise_latex(filename,caption,row_name,col_name, ...
    row_values,col_values,recon_text,rv_text) % Generated with AI assistance

% This 
fid = fopen(filename,'w');

fprintf(fid,'%s\n', ...
    '\begin{sidewaystable}',...
    '\centering', ...
    ['\caption{',caption,'}'], ...
    ['\begin{tabular}{l',repmat('c',1,numel(col_values)),'}'], ...
    '\hline');

headers = [string(row_name), ...
    reshape(compose('%s = %g',col_name,col_values),1,[])];

fprintf(fid,'%s\n',[char(strjoin(headers,' & ')),' \\'],'\hline');

for i = 1:numel(row_values)
    cells = strings(1,numel(col_values));

    for j = 1:numel(col_values)
        recon = strrep(recon_text(i,j),'+/-','$\pm$');
        rv = strrep(rv_text(i,j),'+/-','$\pm$');

        cells(j) = "\begin{tabular}{@{}c@{}}" + ...
            recon + " \\ (" + rv + ")\end{tabular}";
    end

    line = string(row_values(i)) + " & " + ...
        strjoin(cells,' & ') + " \\";
    fprintf(fid,'%s\n',char(line));
end

fprintf(fid,'%s\n', ...
    '\hline','\end{tabular}', '\end{sidewaystable}');
fclose(fid);
end

function matches = alignment_dimensions_match(mdl)
% Mirror the score-row collapsing used by the supplied fbd.opt().
% Its signed-permutation code also assumes equal component counts.
T1 = mdl.X11n*mdl.V11;
T2 = mdl.X22n*mdl.V22;
U1 = uniquetol(T1,1e-6,'ByRows',true);
U2 = uniquetol(T2,1e-6,'ByRows',true);
matches = isequal(size(U1),size(U2)) && ~isempty(U1);
end

function [recon,rv,status] = evaluate_reconstruction(X11,X22,X12,F1,F2,n_perms)
% Fit only observed diagonal blocks; use X12 only to evaluate prediction.
recon = NaN;
rv = NaN;
status = "ok";

% A random split can omit a factor level in one block.
if ~isequal(unique(F1),unique(F2))
    status = "unmatched factor levels";
    return;
end

[~,parglmo1] = parglm(X11-mean(X11,1),F1,'Preprocessing',0);
[~,parglmo2] = parglm(X22-mean(X22,1),F2,'Preprocessing',0);

mdl = fbd(parglmo1,parglmo2,n_perms);
if ~alignment_dimensions_match(mdl)
    status = "alignment dimensions mismatch";
    return;
end
mdl.opt();
mdl.pred_X1X2();

% Preserve the original target: an independently fitted factor matrix.
[~,parglmo12] = parglm(X12-mean(X12,1),F1,'Preprocessing',0);
target = parglmo12.factors{1}.matrix;
prediction = mdl.X1X2n;

if ~isequal(size(target),size(prediction))
    status = "prediction dimensions mismatch";
    return;
end

recon = norm(target-prediction,'fro')^2 / norm(target,'fro')^2;
rv = rv_coefficient(target,prediction);

if ~isfinite(recon) || ~isfinite(rv)
    status = "undefined metric";
    if ~isfinite(recon), recon = NaN; end
    if ~isfinite(rv), rv = NaN; end
end

end

