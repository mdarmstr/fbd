%% Compare blockwise GLM with pooled GLM before splitting
clear all; %#ok
%addpath(genpath('../../MEDA'));
load('split_data.mat');

rng('shuffle');
rng_state = rng;
n_perms = 2; %placeholder for independence testing

n1 = size(trainblk11,1);
p1 = size(trainblk11,2);

F1 = blk1_mta(:,2);
F2 = blk2_mta(:,2);
Fa = [F1; F2];

%% 1 independent, blockwise method
X11 = preprocess2D(trainblk11,'Preprocessing',2);
X22 = preprocess2D(trainblk22,'Preprocessing',2);
X12 = preprocess2D(testblk12,'Preprocessing',2);

[~, parglmo11] = parglm(X11,F1,'Preprocessing',0);
[~, parglmo22] = parglm(X22,F2,'Preprocessing',0);
[~, parglmo12] = parglm(X12,F1,'Preprocessing',0);

mdl_original = fbd(parglmo11,parglmo22,n_perms);
mdl_original.opt();
mdl_original.pred_X1X2();

target_original = parglmo12.factors{1}.matrix;
prediction_original = mdl_original.X1X2n;

error_original = norm(target_original-prediction_original,'fro')^2 / norm(target_original,'fro')^2;

rv_original = rv_coefficient(target_original,prediction_original);

plot_reconstruction(target_original,prediction_original,'Original blockwise GLM');

%% 2 Fit the complete matrix, then split
Xt = [trainblk11, testblk12;
      testblk21, trainblk22];

Xt = preprocess2D(Xt,'Preprocessing',2);

[~, parglmo_full] = parglm(Xt,Fa,'Preprocessing',0);
Xtr = parglmo_full.factors{1}.matrix;

X11r = Xtr(1:n1,1:p1);
X12r = Xtr(1:n1,p1+1:end);
X22r = Xtr(n1+1:end,p1+1:end);

[~, parglmo11r] = parglm(X11r,F1,'Preprocessing',0);
[~, parglmo22r] = parglm(X22r,F2,'Preprocessing',0);

mdl_pooled = fbd(parglmo11r,parglmo22r,n_perms);
mdl_pooled.opt();
mdl_pooled.pred_X1X2();

target_pooled = X12r;
prediction_pooled = mdl_pooled.X1X2n;

error_pooled = norm(target_pooled-prediction_pooled,'fro')^2 / norm(target_pooled,'fro')^2;

rv_pooled = rv_coefficient(target_pooled,prediction_pooled);

plot_reconstruction(target_pooled,prediction_pooled, 'Pooled GLM before splitting');

%% Results
Method = ["Original blockwise GLM"; "Pooled GLM before splitting"];
RelativeSquaredError = [error_original; error_pooled];
RV = [rv_original; rv_pooled];

results = table(Method,RelativeSquaredError,RV);
disp(results);

write_latex(results,'fbd_comparison.tex');

%% Helper functions
function plot_reconstruction(target,prediction,name)
    figure('Name',name);

    limits = [min([target(:); prediction(:)]), ...
              max([target(:); prediction(:)])];

    subplot(3,1,1);
    imagesc(target);
    clim(limits);
    colorbar;
    title('Nom');

    subplot(3,1,2);
    imagesc(prediction);
    clim(limits);
    colorbar;
    title('Rec');

    difference = target-prediction;
    limit = max(abs(difference(:)));

    subplot(3,1,3);
    imagesc(difference);
    clim([-limit,limit]);
    colorbar;
    title('diff');
end

function write_latex(results,filename) %% SIGNIFICANT AI ASSISTANCE
    fid = fopen(filename,'w');

    fprintf(fid,'%s\n', ...
        '\begin{table}[ht]', ...
        '\centering', ...
        ['\caption{FBD reconstruction of the off-diagonal block. ' ...
         'Each method uses its own factor target. The pooled method ' ...
         'uses the complete matrix before splitting.}'], ...
        '\label{tab:fbd-comparison}', ...
        '\begin{tabular}{lrr}', ...
        '\hline', ...
        'Method & Relative squared error & RV \\', ...
        '\hline');

    for i = 1:height(results)
        fprintf(fid,'%s & %.3g & %.3g %s\n', ...
            char(results.Method(i)), ...
            results.RelativeSquaredError(i),results.RV(i),'\\');
    end

    fprintf(fid,'%s\n','\hline','\end{tabular}','\end{table}');
    fclose(fid);
end