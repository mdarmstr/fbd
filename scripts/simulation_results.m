%% SCRIPT: Repeated Positive & Negative Case Analysis
%  This script replicates the Positive and Negative case generation 
%  N times, collects the probability (p-value) for each run, and
%  plots the distribution of these p-values for each case.

% Ensure you have all the required functions in your path, including:
%  create_design, simuleMV, blockDiagonalSampling, parglm, fbd, etc.

clear; close all; clc;

%% Parameters
N = 2000;                 % Number of simulations
reps = 10;
vars = 300;
levels = {[1, 2, 3, 4], [1, 2, 3]};
rng('shuffle');         % Shuffle RNG seed for each run

% Arrays to store p-values
pVals_Positive = zeros(N, 1);
pVals_Negative = zeros(N, 1);

%% Loop through N simulations
for i = 1:N
    
    % ------------------ POSITIVE CASE ------------------
    F = create_design(levels, 'Replicates', reps);
    disp(size(F))
    X = zeros(size(F,1), vars);

    % Prepare random effect levels
    fi = cell(1, length(levels{1}));
    fj = cell(1, length(levels{2}));
    for ii = 1:length(levels{1})
        fi{ii} = randn(1, vars);
    end
    for jj = 1:length(levels{2})
        fj{jj} = randn(1, vars);
    end

    % Generate data, incorporate "fi" and "fj" offsets
    for ii = 1:length(levels{1})
        for jj = 1:length(levels{2})
            idx = (F(:,1) == levels{1}(ii)) & (F(:,2) == levels{2}(jj));
            X(idx,:) = simuleMV(sum(idx), vars, 'LevelCorr', 8) + ...
                repmat(fi{ii} + fj{jj}, sum(idx), 1);
        end
    end

    % Shuffle row order
    rp = randperm(size(X,1));
    X = X(rp,:);
    F = F(rp,:);

    % Split data into X1, X2
    [X1, X2] = blockDiagonalSampling(X, 0.4, 'rows');
    F1 = F(1:size(X1,1), :);
    F2 = F(size(X1,1)+1:end, :);

    % Fit parglm
    [~, parglmo1] = parglm(X1, F1, 'Preprocessing', 1);
    [~, parglmo2] = parglm(X2, F2, 'Preprocessing', 1);

    % Run fbd
    [p_Pos, ~, ~, ~] = fbd(parglmo1, parglmo2, F1, F2, 1, 1000);

    pVals_Positive(i) = p_Pos;
    fprintf('Postive simulation %d complete\n',i)

    % ------------------ NEGATIVE CASE ------------------
    % Regenerate data for negative scenario
    F = create_design(levels, 'Replicates', reps);
    X = zeros(size(F,1), vars);

    % Prepare random effect levels again
    fi = cell(1, length(levels{1}));
    fj = cell(1, length(levels{2}));
    for ii = 1:length(levels{1})
        fi{ii} = randn(1, vars);
    end
    for jj = 1:length(levels{2})
        fj{jj} = randn(1, vars);
    end

    for ii = 1:length(levels{1})
        for jj = 1:length(levels{2})
            idx = (F(:,1) == levels{1}(ii)) & (F(:,2) == levels{2}(jj));
            X(idx,:) = simuleMV(sum(idx), vars, 'LevelCorr', 8) + ...
                repmat(fi{ii} + fj{jj}, sum(idx), 1);
        end
    end
    X2_neg = X;  % entire dataset for negative
    F2_neg = F;
    
    % Reuse the same X1, F1 from above as the "first dataset" 
    % (for demonstration) or you can regenerate X1, F1 if desired.
    
    % For example, we can keep X1, F1 from above to test a "no-change" scenario:
    [~, parglmo1_neg] = parglm(X1, F1, 'Preprocessing', 1);
    [~, parglmo2_neg] = parglm(X2_neg, F2_neg, 'Preprocessing', 1);

    [p_Neg, ~, ~, ~] = fbd(parglmo1_neg, parglmo2_neg, F1, F2_neg, 1, 1000);

    pVals_Negative(i) = p_Neg;
    fprintf('Negative simulation %d complete\n',i)

end

%% ------------------ Plot results ------------------
figure('Name','Distribution of p-values across simulations','Color','w');
hold on; box on;
histogram(pVals_Positive, 'BinWidth', 0.01,'FaceColor','auto');
xlabel('p-value');
ylabel('Frequency');

box on;
histogram(pVals_Negative, 'BinWidth', 0.01,'FaceColor','auto');
title('Negative Case (red) vs Positive case (blue)');
xlabel('p-value');
ylabel('Frequency');
hold off;

% % Optionally, you can show a summary boxplot of the p-values:
% figure('Name','Boxplots of p-values','Color','w');
% boxplot([pVals_Positive, pVals_Negative], 'Labels', {'Positive','Negative'});
% ylabel('p-value');
% title('Boxplot of p-values across N simulations');

%% ------------------ Helper Functions ------------------

function [block1, block2] = blockDiagonalSampling(X, p, mode)
% blockDiagonalSampling Subsets a block diagonal sampling from a matrix.
%
%   [block1, block2] = blockDiagonalSampling(X, p, mode) extracts two
%   rectangular submatrices from the input matrix X according to the
%   specified sampling percentage p. The parameter 'mode' determines how
%   the matrix is partitioned:
%
%       'rows'  - Split by rows only: the first block takes the first 
%                 round(m*p) rows (all columns), and the second block uses
%                 the remaining rows.
%
%       'cols'  - Split by columns only: the first block takes the first 
%                 round(n*p) columns (all rows), and the second block uses
%                 the remaining columns.
%
%       'both'  - Split both rows and columns: the first block is the top
%                 left submatrix with round(m*p) rows and round(n*p) columns,
%                 and the second block is the bottom right submatrix using
%                 the remaining rows and columns.
%
%   Inputs:
%       X    - The input m x n matrix.
%       p    - Sampling percentage (0 < p < 1). E.g., 0.3 means 30%.
%       mode - A string with options: 'rows', 'cols', or 'both'.
%
%   Outputs:
%       block1 - The submatrix corresponding to the top left block.
%       block2 - The submatrix corresponding to the bottom right block.
%

    if nargin < 3
        mode = 'both'; % Default mode if not provided.
    end

    [m, n] = size(X);

    switch lower(mode)
        case 'rows'
            % Determine number of rows for block 1
            r1 = round(m * p);
            block1 = X(1:r1, :);        % Top rows (all columns)
            block2 = X(r1+1:end, :);      % Remaining rows (all columns)
            
        case 'cols'
            % Determine number of columns for block 1
            c1 = round(n * p);
            block1 = X(:, 1:c1);        % Left columns (all rows)
            block2 = X(:, c1+1:end);      % Remaining columns (all rows)
            
        case 'both'
            % Determine both rows and columns for block 1
            r1 = round(m * p);
            c1 = round(n * p);
            block1 = X(1:r1, 1:c1);      % Top left block
            block2 = X(r1+1:end, c1+1:end); % Bottom right block
            
        otherwise
            error('Unknown mode. Please use ''rows'', ''cols'', or ''both''.');
    end
end


function [p,T1oe,T1r,T2oe] = fbd(parglmoA, parglmoB, F1, F2, fctrs, n_perms)

X1 = parglmoA.data;
X2 = parglmoB.data;

X1ne = parglmoA.residuals;
X2ne = parglmoB.residuals;

X1n = parglmoA.factors{fctrs(1)}.matrix;
X2n = parglmoB.factors{fctrs(1)}.matrix;

D1 = parglmoA.D(:,parglmoA.factors{fctrs(1)}.Dvars);
D2 = parglmoB.D(:,parglmoB.factors{fctrs(1)}.Dvars);

% Scores calculation

[U1n, S1n, V1] = svds(X1n, rank(X1n));
[U2n, S2n, V2] = svds(X2n, rank(X2n));

[V1,T] = rotatefactors(V1,'Method','varimax','maxit',5000,'reltol',1e-6);
V2 = rotatefactors(V2,'Method','varimax','maxit',5000,'reltol',1e-6);

T1o = X1n * V1;
T2o = X2n * V2;

% %Varimax rotation on the scores
% T1o = U1n * S1n;
% T1o = varimax(T1o,false);
% T2o = U2n * S2n;
% T2o = varimax(T2o,false);

% Procrustes rotation as a separate script
R = procrustes_rotation(T1o,T2o,F1,F2);

% Incorporate noise and apply the rotation
T1oe = ((X1n + X1ne) * V1);    % T1 with noise (before rotation)
T1r = T1oe * R;         % T1 after rotation
T2oe = (X2n + X2ne) * V2;        % T2 after noise (no rotation)

%Calculate diasmetic statistic
[~,Fd] = closestSignedPermutation(R);

%Permutation test

Fp = zeros([1,n_perms]);

for ii = 1:n_perms
    perms = randperm(size(X1,1)); % permuted data (permute whole data matrix)
    Xperm = X1(perms, :);
    pD1 =  pinv(D1'*D1)*D1';
    Bperm = pD1*Xperm;
    X1perm = D1*Bperm;
    [~,~,Vpm] = svds(X1perm,rank(X1perm));
    %Vpm = rotatefactors(Vpm,'Method','varimax','maxit',1000); %does not
    %always converge to a useful solution
    Tpm = X1perm * Vpm * T;
    R = procrustes_rotation(Tpm,T2o,F1,F2);
    [~, err] = closestSignedPermutation(R);
    Fp(ii) = err;
end

p = (sum(Fp < Fd) + 1) / (n_perms + 1);

end

function R = procrustes_rotation(T1o,T2o,F1,F2)

% Rotate T1o to fit T2o

% [U1n, S1n, V1] = svds(X1n, rank(X1n));
% [U2n, S2n, V2] = svds(X2n, rank(X2n));
% 
% T1o = U1n * S1n;
% T2o = U2n * S2n;

[T1u, ord1, ~] = uniquetol(T1o, 'ByRows', true, 'PreserveRange', true);
[T2u, ord2, ~] = uniquetol(T2o, 'ByRows', true, 'PreserveRange', true);

lvls1 = F1(ord1, 1);
lvls2 = F2(ord2, 1);

[~, perm_idx] = ismember(lvls1, lvls2);
n = numel(lvls1);
P = eye(n);
P = P(perm_idx, :);
T2u = P * T2u;

M = T1u' * T2u;
[Up, ~, Vp] = svd(M);
R = Up * Vp';  % Rotation matrix

end

function [P, froError] = closestSignedPermutation(R)
% closestSignedPermutation computes the closest signed permutation matrix
% to a given rotation matrix R, allowing for reflections (i.e., entries of -1).
%
%   [P, froError] = closestSignedPermutation(R)
%
% Inputs:
%   R - an n-by-n rotation matrix.
%
% Outputs:
%   P - the closest signed permutation matrix to R.
%   froError - the Frobenius norm of the difference, ||R - P||_F.
%
% The algorithm works by solving an assignment problem to maximize the sum
% of the absolute values of the selected entries in R. It uses MATLAB's 
% matchpairs function (available in the Statistics and Machine Learning Toolbox)
% to find the optimal assignment.

    n = size(R, 1);
    
    % Form the cost matrix as the negative absolute value of R
    costMat = -abs(R);
    
    % Solve the assignment problem using matchpairs (Hungarian algorithm)
    % The second argument (-Inf) ensures a complete assignment.
    assignment = matchpairs(costMat, 1e6);
    
    % Initialize the signed permutation matrix P
    P = zeros(n);
    
    % For each assignment, set the corresponding entry in P to the sign of R
    for k = 1:size(assignment, 1)
        i = assignment(k, 1);
        j = assignment(k, 2);
        if R(i, j) >= 0
            P(i, j) = 1;
        else
            P(i, j) = -1;
        end
    end
    
    % Calculate the Frobenius norm of the difference
    froError = norm(R - P, 'fro')^2 / norm(R,'fro')^2;
end

function [P2, froError2] = nextClosestSignedPermutation(R)
% nextClosestSignedPermutation computes the next closest signed permutation matrix
% to the given rotation matrix R.
%
%   [P2, froError2] = nextClosestSignedPermutation(R)
%
% It works by first computing the optimal assignment, then, for each assignment
% element, temporarily disallowing that match and recomputing the assignment.
% The candidate with the lowest Frobenius error (that is larger than the optimal error)
% is chosen as the next closest permutation.
%
    n = size(R, 1);
    % Original cost matrix
    costMatOrig = -abs(R);
    
    % Get the optimal assignment and error
    [P_opt, froError_opt] = closestSignedPermutation(R);
    optAssignment = matchpairs(costMatOrig, 1e6);
    
    bestError = Inf;
    bestP = [];
    
    % Loop over each element in the optimal assignment
    for k = 1:size(optAssignment, 1)
        costMat = costMatOrig;
        i = optAssignment(k, 1);
        j = optAssignment(k, 2);
        % Force this match to be unavailable by setting its cost to a very high value
        costMat(i, j) = 1e6;
        
        % Solve the assignment problem again
        candidateAssignment = matchpairs(costMat, 1e6);
        % Build candidate permutation matrix
        candidateP = zeros(n);
        for l = 1:size(candidateAssignment, 1)
            r_i = candidateAssignment(l, 1);
            r_j = candidateAssignment(l, 2);
            if R(r_i, r_j) >= 0
                candidateP(r_i, r_j) = 1;
            else
                candidateP(r_i, r_j) = -1;
            end
        end
        candidateError = norm(R - candidateP, 'fro');
        
        % Check if candidate error is higher than optimal and lower than best found so far
        if candidateError > froError_opt && candidateError < bestError
            bestError = candidateError;
            bestP = candidateP;
        end
    end
    
    if isempty(bestP)
        % If no candidate found that is worse than the optimal, return the optimal.
        bestP = P_opt;
        bestError = froError_opt;
    end
    
    P2 = bestP;
    froError2 = bestError;
end

function [F,parglmo1,parglmo2] = return_parglmo_positive(dataset_number, reps, vars, levels, col_mat_table)


F = create_design(levels, 'Replicates', reps);
X = zeros(size(F,1),vars);

for i = 1:length(levels{1})
    X(find(F(:,1) == levels{1}(i)),:) = simuleMV(length(find(F(:,1) == levels{1}(i))),vars,8) + repmat(randn(1,vars),length(find(F(:,1) == levels{1}(i))),1);
end

% Create a block diagonal mask to split the data into two groups
A1 = ones(50,30);
A2 = 2.*ones(40,40);
msk = blkdiag(A1,A2);

% Group 1 (msk == 1)
[r1, c1] = find(msk == 1);
row_range1 = min(r1):max(r1);
col_range1 = min(c1):max(c1);
X1s = X1(row_range1, col_range1);
X1es = X1e(row_range1, col_range1);
F1s = F1(row_range1, :);

% Group 2 (msk == 2)
[r1, c1] = find(msk == 2);
row_range2 = min(r1):max(r1);
col_range2 = min(c1):max(c1);
X2s = X1(row_range2, col_range2);
X2es = X1e(row_range2, col_range2);
F2s = F1(row_range2, :);

end

function [Xn, Xe, D, F, parglmo] = create_plots_for_datasets(dataset_number, reps, vars, levels, col_mat_table)
    % Create design matrix F (assumes create_design is provided in MEDA toolbox)
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
    % Populate X with simulated data using simuleMV (assumed to be part of MEDA)
    for i = 1:length(levels{1})
        for j = 1:length(levels{2})
            idx = find(F(:, 1) == levels{1}(i) & F(:, 2) == levels{2}(j));
            X(idx, :) = simuleMV(reps, vars, 'LevelCorr', 8) + repmat(fi{i} + fj{j}, reps, 1);
        end
    end
    
    % Run parglm analysis (assumed to be part of MEDA)
    [~, parglmo] = parglm(X, F, 'Preprocessing', 1);
    
    % Aggregate the factors (using only the first factor)
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
    
    % Plot matrices (these functions save images and then close the figure)
    plot_matrix(Xn, col_mat_table, ' ', ' ', ['X', num2str(dataset_number)]);
    plot_matrix(parglmo.D, col_mat_table, ' ', ' ', ['D', num2str(dataset_number)]);
    plot_matrix(parglmo.Xnan, col_mat_table, ' ', ' ', ['X', num2str(dataset_number), 'e']);
    plot_matrix(parglmo.B, col_mat_table, ' ', ' ', ['B', num2str(dataset_number)]);
end

function plot_matrix(mat, col_map, is_y, is_x, filename)
    % Normalize matrix for imagesc display
    imagesc(mat ./ max(abs(mat), [], 'all'));
    colormap(col_map);
    [numRows, numCols] = size(mat);
    
    % Calculate figure size (using a fixed dpi)
    dpi = 15;
    figWidthInches = numCols / dpi;
    figHeightInches = numRows / dpi;
    set(gcf, 'Units', 'inches', 'Position', [1, 1, figWidthInches, figHeightInches]);
    
    % Set labels and formatting
    xlabel(is_x, 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'k');
    ylabel(is_y, 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'k');
    set(gca, 'XTick', []); set(gca, 'YTick', []);
    set(gca, 'XAxisLocation', 'top'); set(gca, 'box', 'off');
    
    % Ensure output directory exists
    if ~exist('figures', 'dir')
        mkdir('figures');
    end
    exportgraphics(gcf, fullfile('figures', [filename, '.png']), 'BackgroundColor', 'none');
    close(gcf);
end

function interpolatedColors = interpolateColors(color1, color2, numPoints)
    t = linspace(0, 1, numPoints);
    interpolatedColors = zeros(length(t), 3);
    for i = 1:length(t)
        interpolatedColors(i, :) = t(i) * color2 + (1 - t(i)) * color1;
    end
end
