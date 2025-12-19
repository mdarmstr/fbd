clear; close all; clc;

% settings
reps_pos   = 40;
smpl_ratio = 0.4;
N          = 100;

designs = { [1 2 3], [1 2 3 4], [1 2 3 4 5] };

% compute first (because fbd_demo has close all;)
ps_pos_all = cell(1,3);
ps_neg_all = cell(1,3);

for j = 1:3
    levels = {designs{j}};
    fprintf('\n=== Running design %s ===\n', mat2str(designs{j}));
    [ps_pos_all{j}, ps_neg_all{j}] = fbd_demo(levels, reps_pos, smpl_ratio, N);
end

% plot

set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');

figure('Color','w','Units','inches','Position',[1 1 16 8], ...
       'Name','FBD demo: histogram + empirical CDF');
t = tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

for j = 1:3
    axH = nexttile(t, j);
    axC = nexttile(t, j+3);

    plot_fbd_panel(axH, axC, ps_pos_all{j}, ps_neg_all{j});

    % column heading goes on the top axes
    title(axH, sprintf('Design: %s', mat2str(designs{j})));
end

title(t, sprintf('FBD demo (reps=%d, smpl\\_ratio=%.2f, N=%d)', reps_pos, smpl_ratio, N));
