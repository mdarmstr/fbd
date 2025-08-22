% Add the MEDA toolbox to the path (ensure MEDA toolbox is installed)
addpath('MEDA/');

% Define colors
red = rgb2hsv([202, 45, 48] ./ 255);
tel = rgb2hsv([0, 140, 186] ./ 255);
blk = [52, 53, 54] ./ 255;
col_mat_table = hsv2rgb(interpolateColors(red, tel, 100));

% Generate and plot datasets
%[X1,X1e,D1,F1] = create_plots_for_datasets(1, 10, 70, {[1, 2, 3], [1, 2, 3]}, col_mat_table);

rp = randperm(size(X1,1));

X1 = X1(rp,:);
X1e = X1e(rp,:);
F1 = F1(rp,:);

A1 = ones(50,30);
A2 = 2.*ones(40,40);

msk = blkdiag(A1,A2);

[r1,c1] = find(msk == 1);

row_range1 = min(r1):max(r1);
col_range1 = min(c1):max(c1);

X1s = X1(row_range1,col_range1);
X1es = X1e(row_range1,col_range1);
F1s = F1(row_range1,:);

[r1,c1] = find(msk == 2);

row_range2 = min(r1):max(r1);
col_range2 = min(c1):max(c1);

X2s = X1(row_range2,col_range2);
X2es = X1e(row_range2,col_range2);
F2s = F1(row_range2,:);


% Calculate projections
% For dataset 1 (T1) and dataset 2 (T2) before rotation:
[U1,S1,V1] = svds(X1s, rank(X1s));
[U2,S2,V2] = svds(X2s, rank(X2s));

T1_orig = U1 * S1;  % T1 before rotation (from X1)
T2_orig = U2 * S2;  % T2 before rotation (from X2)

% Remove duplicate rows if desired
[T1u,ord1,~] = uniquetol(T1_orig, 'ByRows', true,'PreserveRange',true);
[T2u,ord2,~] = uniquetol(T2_orig, 'ByRows', true,'PreserveRange',true);

lvls1 = F1s(ord1,1);
lvls2 = F2s(ord2,1);

[~, perm_idx] = ismember(lvls1, lvls2);
% perm_idx = [2; 3; 1]  => means: row 1 of lvls1 is at row 2 of lvls2, etc.

% Step 2: Build permutation matrix
n = numel(lvls1);
P = eye(n);
P = P(perm_idx, :);   % permute the identity matrix

T2u = P*T2u;

% Compute the outer product between the unique rows and find the rotation
M = T1u' * T2u;
[U_tmp, ~, V_tmp] = svd(M);
R = U_tmp * V_tmp';  % Optimal rotation from Procrustes

% For "after rotation" we incorporate noise and then rotate T1
T1_noisy = ((X1s + X1es) * V1);  % T1 with noise (before rotation)
T1_rot = T1_noisy * R;          % T1 after applying the optimal rotation

T2_noisy = (X2s + X2es) * V2;      % T2 after adding noise (no rotation is applied)

% Extract the first two dimensions for plotting
T1_orig_data = T1_orig(:, 1:2);
T2_orig_data = T2_orig(:, 1:2);

T1_rot_data = T1_rot(:, 1:2);
T2_noisy_data = T2_noisy(:, 1:2);

%% === Prepare Class Labels and Colors ===

% Assuming F1 and F2 contain the class labels in their first column:
ObsClass1 = F1(row_range1,1);
ObsClass2 = F1(row_range2,1);

% Use the union of classes from both datasets to ensure consistency
classes = unique([ObsClass1; ObsClass2]);
colors = lines(length(classes));  % Use distinct colors

%% === Create Figure with Subplots ===

figure;

% --- Subplot 1: Before Rotation ---
subplot(1,2,1);
hold on;
title('Before Rotation - Positive');
xlabel('Component 1');
ylabel('Component 2');

% Plot T1_orig (filled markers) by class
for i = 1:length(classes)
    idx = ObsClass1 == classes(i);
    scatter(T1_noisy(idx,1), T1_noisy(idx,2), 50, colors(i,:), 'o', 'filled');
end

% Plot T2_orig (hollow markers) by class
for i = 1:length(classes)
    idx = ObsClass2 == classes(i);
    scatter(T2_noisy(idx,1), T2_noisy(idx,2), 50, colors(i,:), 'o');
end

legend(arrayfun(@(c) ['Level ' num2str(c)], classes, 'UniformOutput', false), 'Location', 'best');
hold off;

% --- Subplot 2: After Rotation ---
subplot(1,2,2);
hold on;
title('After Rotation - Positive');
xlabel('Component 1');
ylabel('Component 2');

% Plot rotated T1 (filled markers) by class (using ObsClass1)
for i = 1:length(classes)
    idx = ObsClass1 == classes(i);
    scatter(T1_rot_data(idx,1), T1_rot_data(idx,2), 50, colors(i,:), 'o', 'filled');
end

% Plot noisy T2 (hollow markers) by class (using ObsClass2)
for i = 1:length(classes)
    idx = ObsClass2 == classes(i);
    scatter(T2_noisy_data(idx,1), T2_noisy_data(idx,2), 50, colors(i,:), 'o');
end

legend(arrayfun(@(c) ['Level ' num2str(c)], classes, 'UniformOutput', false), 'Location', 'best');
hold off;

%% --- Formatting both subplots ---
for ax = findall(gcf,'Type','axes')'
    set(ax, 'FontName', 'Arial', ...
            'FontSize', 14, ...
            'LineWidth', 1, ...
            'Box', 'off');
    ax.XLabel.FontSize = 15;
    ax.YLabel.FontSize = 15;
    ax.Title.FontSize  = 15;
end

% Legend formatting
lgd = findall(gcf,'Type','Legend');
set(lgd, 'FontSize', 12, 'Box', 'off');

% Save as cropped, vector PDF
exportgraphics(gcf,'rotation_subplots.pdf','ContentType','vector');

%% Helper Functions

% Function to create plots for all datasets
function [Xn,Xe,D,F] = create_plots_for_datasets(dataset_number, reps, vars, levels, col_mat_table)
    % Generate design matrix F using create_design (external function)
    F = create_design(levels, 'Replicates', reps);
    writematrix(F, ['F', num2str(dataset_number), '.csv']);
    
    % Initialize data matrix X
    X = zeros(size(F, 1), vars);
    
    % Create level means
    fi = cell(1, length(levels{1}));
    fj = cell(1, length(levels{2}));
    for i = 1:length(levels{1})
        fi{i} = randn(1, vars);
    end
    for j = 1:length(levels{2})
        fj{j} = randn(1, vars);
    end
    
    % Populate X with simulated data using simuleMV (external function)
    for i = 1:length(levels{1})
        for j = 1:length(levels{2})
            idx = find(F(:, 1) == levels{1}(i) & F(:, 2) == levels{2}(j));
            X(idx, :) = simuleMV(reps, vars, 'LevelCorr', 7) + repmat(fi{i} + fj{j}, reps, 1);
        end
    end
    
    % Perform parglm (external function)
    [~, parglmo] = parglm(X, F,'Preprocessing',1);
    
    % Create new matrices - one factor now
    Xn = zeros(size(X, 1), size(X, 2));
    for ii = 1:1
        Xn = Xn + parglmo.factors{ii}.matrix;
    end
    
    Xe = parglmo.Xnan;
    D = parglmo.D;

    % Write matrices to CSV files
    writematrix(Xn, ['X', num2str(dataset_number), '.csv']);
    writematrix(parglmo.D, ['D', num2str(dataset_number), '.csv']);
    writematrix(parglmo.Xnan, ['X', num2str(dataset_number), 'e.csv']);
    
    % Plot matrices
    plot_matrix(Xn, col_mat_table, ' ', ' ', ['X', num2str(dataset_number)]);
    plot_matrix(parglmo.D, col_mat_table, ' ', ' ', ['D', num2str(dataset_number)]);
    plot_matrix(parglmo.Xnan, col_mat_table, ' ', ' ', ['X', num2str(dataset_number), 'e']);
    plot_matrix(parglmo.B, col_mat_table, ' ', ' ', ['B', num2str(dataset_number)]);
end

% Define the plot_matrix function
function plot_matrix(mat, col_map, is_y, is_x, filename)
    % Normalize the matrix for imagesc
    imagesc(mat ./ max(abs(mat), [], 'all'));
    
    % Set the custom colormap
    colormap(col_map);
    
    % Get the size of the matrix
    [numRows, numCols] = size(mat);

    % Calculate the appropriate figure size in pixels
    dpi = 15; % Dots per inch for screen display, adjust as necessary
    
    % Calculate figure dimensions in inches
    figWidthInches = numCols / dpi;
    figHeightInches = numRows / dpi;

    % Set the figure size to match the image size
    set(gcf, 'Units', 'inches', 'Position', [1, 1, figWidthInches, figHeightInches]);

    % Add labels with larger, bold text
    xlabel(is_x, 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'k');
    ylabel(is_y, 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'k');

    % Remove x and y axis tick labels
    set(gca, 'XTick', []);
    set(gca, 'YTick', []);

    % Position the x label at the top
    set(gca, 'XAxisLocation', 'top');

    % Remove the border
    set(gca, 'box', 'off');
    
    % Ensure the 'figures' directory exists
    if ~exist('figures', 'dir')
        mkdir('figures');
    end

    % Save the figure
    exportgraphics(gcf, fullfile('figures', [filename, '.png']), 'BackgroundColor', 'none');

    close all;
end

% Define the interpolateColors function
function interpolatedColors = interpolateColors(color1, color2, numPoints)
    % interpolateColors creates a linear interpolation between two colors.
    %
    % Inputs:
    %   color1 - 1x3 vector representing the first RGB color
    %   color2 - 1x3 vector representing the second RGB color
    %   numPoints - The number of points in the interpolation
    %
    % Output:
    %   interpolatedColors - numPoints x 3 matrix of interpolated RGB colors

    % Create a linspace vector for interpolation
    t = linspace(0, 1, numPoints); % numPoints from 0 to 1

    % Initialize the matrix to hold the interpolated colors
    interpolatedColors = zeros(length(t), 3);

    % Perform the interpolation
    for i = 1:length(t)
        interpolatedColors(i, :) = t(i) * color2 + (1 - t(i)) * color1;
    end
end
