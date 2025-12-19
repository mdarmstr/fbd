function [ps_pos, ps_neg] = fbd_demo(levels,reps_pos,smpl_ratio, N)

% DEMONSTRATION OF FBD  
close all;
% Arrays to store p-values
ps_pos = zeros(N, 1);
ps_neg = zeros(N, 1);
%levels = {[1, 2, 3]};
nse = 1;
%reps_pos = 40;
vars_pos = 300;
%N = 100;

rng('shuffle');

%% Loop through N simulations
for i = 1:N

% Positive
[X1, X2, F1, F2, szPos] = simul_data('pos', levels, reps_pos, vars_pos, nse, smpl_ratio, 'both');

[~, parglmo1_pos] = parglm(X1 - mean(X1), F1, 'Preprocessing', 0); %Force the coding matrix to ignore column of ones.
[~, parglmo2_pos] = parglm(X2 - mean(X2), F2, 'Preprocessing', 0);

pos = fbd(parglmo1_pos,parglmo2_pos,2000);
pos.test()
p_pos = pos.p;

% Negative
[X1_neg, X2_neg, F1_neg, F2_neg, szNeg] = simul_data('neg', levels, reps_pos, vars_pos, nse, smpl_ratio, 'both');

[~, parglmo1_neg] = parglm(X1_neg - mean(X1_neg), F1_neg, 'Preprocessing', 0);
[~, parglmo2_neg] = parglm(X2_neg - mean(X2_neg), F2_neg, 'Preprocessing', 0);

neg = fbd(parglmo1_neg,parglmo2_neg,2000);
neg.test()
p_neg = neg.p;
%disp(szPos); disp(szNeg);

fprintf('Simulation %d complete\n',i)

ps_pos(i) = p_pos;
ps_neg(i) = p_neg;
end

end
%%--------------------------HELPER FUNCTIONS-------------------------------
function [X1, X2, F1, F2, sz] = simul_data(mode, levels, reps_pos, vars_pos, nse, splitFrac, splitMode)
%SIMUL_DATA  Simulate data for FBD experiments (positive/negative cases).
%
%   [X1,X2,F1,F2,sz] = simul_data('pos', levels, reps_pos, vars_pos, nse, 0.4, 'both');
%   [X1,X2,F1,F2,sz] = simul_data('neg', levels, reps_pos, vars_pos, nse, 0.4, 'both');

    if nargin < 7 || isempty(splitMode), splitMode = 'both'; end
    if nargin < 6 || isempty(splitFrac), splitFrac = 0.4; end

    mode = lower(string(mode));
    if mode == "positive", mode = "pos"; end
    if mode == "negative", mode = "neg"; end

    switch mode
        case "pos"
            % ---- simulate one big matrix, shuffle, then blockDiagonalSampling ----
            [X, F] = local_sim_one(levels, reps_pos, vars_pos, nse);

            % Shuffle row order
            rp = randperm(size(X,1));
            X = X(rp,:);
            F = F(rp,:);

            % Split data into X1, X2
            [X1, X2] = blockDiagonalSampling(X, splitFrac, splitMode);
            F1 = F(1:size(X1,1), :);
            F2 = F(size(X1,1)+1:end, :);

            sz = struct();
            sz.mode = 'pos';
            sz.X_full = size(X);
            sz.F_full = size(F);
            sz.X1 = size(X1);  sz.X2 = size(X2);
            sz.F1 = size(F1);  sz.F2 = size(F2);

        case "neg"
            % ---- simulate X1 and X2 independently (exactly like your code path) ----
            reps1 = floor(splitFrac * reps_pos);
            vars1 = floor(splitFrac * vars_pos);

            [X1, F1] = local_sim_one(levels, reps1, vars1, nse);

            reps2 = floor((1 - splitFrac) * reps_pos);
            vars2 = floor((1 - splitFrac) * vars_pos);

            [X2, F2] = local_sim_one(levels, reps2, vars2, nse);

            sz = struct();
            sz.mode = 'neg';
            sz.X1 = size(X1);  sz.X2 = size(X2);
            sz.F1 = size(F1);  sz.F2 = size(F2);
            sz.reps1 = reps1;  sz.vars1 = vars1;
            sz.reps2 = reps2;  sz.vars2 = vars2;

        otherwise
            error('simul_data:badMode', 'mode must be ''pos'' or ''neg''.');
    end
end

% ----------------------- local helper -----------------------
function [X, F] = local_sim_one(levels, reps, vars, nse)
%Prevent data leakage, by wrapper function
    F = createDesign(levels,'Replicates',reps);
    X = zeros(size(F,1),vars);
    for ii = 1:length(levels{1})
        X(find(F(:,1) == levels{1}(ii)),:) = randn(length(find(F(:,1) == levels{1}(ii))),vars) ...
            + nse.*repmat(randn(1,vars),length(find(F(:,1) == levels{1}(ii))),1);
    end
end


function [block1, block2] = blockDiagonalSampling(X, p, mode)
% blockDiagonalSampling Subsets a block diagonal sampling from a matrix.
% REFACTORED AUG 2025 - 'rows' no longer corresponds to an intermediate calculation
% to describe the proportion of data to be sampled. This caused a
% significant error.
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