function fbd_demo(N,levels,nse)
%% TODO
% Drop levels, fewer replicates
% NEVER simulate what's going on in the residuals
% Fix loop error

%% Parameters
%N = 100;           

% Arrays to store p-values
pVals_Positive = zeros(N, 1);
pVals_Negative = zeros(N, 1);
%levels = {[1, 2, 3]};
%nse = 1;
rng('default');

%% Loop through N simulations
for i = 1:N
    
    % ------------------ POSITIVE CASE ------------------
    reps_pos = 40;
    vars_pos = 300;

    % Prepare random effect levels
    F = createDesign(levels,'Replicates',reps_pos);
    X = zeros(size(F,1),vars_pos);
    for ii = 1:length(levels{1})
        X(find(F(:,1) == levels{1}(ii)),:) = randn(length(find(F(:,1) == levels{1}(ii))),vars_pos) + nse.*repmat(randn(1,vars_pos),length(find(F(:,1) == levels{1}(ii))),1);
    end

    % Shuffle row order
    rp = randperm(size(X,1));
    X = X(rp,:);
    F = F(rp,:);

    % Split data into X1, X2
    [X1, X2] = blockDiagonalSampling(X, 0.4, 'both');
    F1 = F(1:size(X1,1), :);
    F2 = F(size(X1,1)+1:end, :);
    %X1 = X1 / sqrt(size(X1,2));
    %X2 = X2 / sqrt(size(X2,2));

    %disp(size(X1))
    %disp(size(X2))

    % Fit parglm
    [~, parglmo1] = parglm(X1 - mean(X1), F1, 'Preprocessing', 0);
    [~, parglmo2] = parglm(X2 - mean(X2), F2, 'Preprocessing', 0);

    % ascao1 = asca(parglmo1);
    % ascao2 = asca(parglmo2);
    % 
    % scores(ascao1.factors{1},'ObsClass',F1(:,1),'Title','ASCA1')
    % scores(ascao2.factors{1},'ObsClass',F2(:,1),'Title','ASCA2')


    % Run fbd
    [p_Pos, ~, ~, ~] = fbd(parglmo1, parglmo2, F1, F2, 1, 1000);

    pVals_Positive(i) = p_Pos;
    fprintf('Postive  simulation %d complete\n',i)

    % ------------------ NEGATIVE CASE ------------------
    reps = floor(0.4*reps_pos);
    vars = floor(0.4*vars_pos);

    F = createDesign(levels,'Replicates',reps);

    X = zeros(size(F,1),vars);
    for ii = 1:length(levels{1})
        X(find(F(:,1) == levels{1}(ii)),:) = randn(length(find(F(:,1) == levels{1}(ii))),vars) + nse.*repmat(randn(1,vars),length(find(F(:,1) == levels{1}(ii))),1);
    end

    X1_neg = X;
    F1_neg = F;
    
    %disp(sqrt(size(X1_neg,2)))
    reps = floor(0.6*reps_pos);
    vars = floor(0.6*vars_pos);

    F = createDesign(levels, 'Replicates', reps);
    X = zeros(size(F,1),vars);
    for ii = 1:length(levels{1})
        X(find(F(:,1) == levels{1}(ii)),:) = randn(length(find(F(:,1) == levels{1}(ii))),vars) + nse.*repmat(randn(1,vars),length(find(F(:,1) == levels{1}(ii))),1);
    end

    X2_neg = X;  % entire dataset for negative
    F2_neg = F;

    %X1_neg = X1_neg / sqrt(size(X1_neg,2));
    %X2_neg = X2_neg / sqrt(size(X2_neg,2));

    %disp(size(X1_neg))
    %disp(size(X2_neg))
    % For example, we can keep X1, F1 from above to test a "no-change" scenario:
    [~, parglmo1_neg] = parglm(X1_neg - mean(X1_neg), F1_neg, 'Preprocessing', 0);
    [~, parglmo2_neg] = parglm(X2_neg - mean(X2_neg), F2_neg, 'Preprocessing', 0);

    % ascao1 = asca(parglmo1_neg);
    % ascao2 = asca(parglmo2_neg);
    % 
    % scores(ascao1.factors{1},'ObsClass',ascao1.design(:,1),'Title','ASCA1')
    % scores(ascao2.factors{1},'ObsClass',ascao2.design(:,1),'Title','ASCA1')

    [p_Neg, ~, ~, ~] = fbd(parglmo1_neg, parglmo2_neg, F1_neg, F2_neg, 1, 1000);

    pVals_Negative(i) = p_Neg;
    fprintf('Negative simulation %d complete\n',i)

end

%% ------------------ Plot results ------------------
figure('Name','Distribution of p-values across simulations','Color','w');
hold on; box on;

% Combine both vectors to determine common bin edges
all_p = [pVals_Positive(:); pVals_Negative(:)];
edges = linspace(min(all_p), max(all_p), 20); % or however many bins you want

% Plot both histograms with the same edges
histogram(pVals_Positive, edges, 'FaceColor', 'b', 'FaceAlpha', 0.5);
histogram(pVals_Negative, edges, 'FaceColor', 'r', 'FaceAlpha', 0.5);

xlabel('p-value');
ylabel('Frequency');
title('Negative Case (red) vs Positive case (blue)');
legend({'Positive','Negative'});
box on;
hold off;

% figure(2)
% cdf_plot(pVals_Positive,pVals_Negative, 'Matrix Distance');
% snapnow;


end

%% ------------------ Helper Functions ------------------

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

            % Shuffle row order (exactly as in your code)
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
% EXACTLY matches the simulation block in your script:
%
%   F = createDesign(levels,'Replicates',reps);
%   X = zeros(size(F,1),vars);
%   for ii = 1:length(levels{1})
%       X(find(F(:,1) == levels{1}(ii)),:) = randn(length(find(F(:,1) == levels{1}(ii))),vars) ...
%           + nse.*repmat(randn(1,vars),length(find(F(:,1) == levels{1}(ii))),1);
%   end

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


function [p,T1oe,T1r,T2oe] = fbd(parglmoA, parglmoB, F1, F2, fctrs, n_perms)

X1 = parglmoA.data;

X2 = parglmoB.data;

F1 = parglmoA.design;
F2 = parglmoB.design;

[F1,idx] = sort(F1,"ascend");
X1 = X1(idx,:);
Z1 = parglmoA.D(idx,:);

[F2,idx] = sort(F2,"ascend");
X2 = X2(idx,:);
Z2 = parglmoB.D(idx,:);

B1hat = pinv(Z1)*X1;
X1n = Z1*B1hat;
E1 = X1 - X1n;

B2hat = pinv(Z2)*X2;
X2n = Z2*B2hat;
E2 = X2 - X2n; 

% Scores calculation
[~, ~, V1] = svds(X1n, rank(X1n));
[~, ~, V2] = svds(X2n, rank(X2n));

T1o = X1n * V1 ;
T2o = X2n * V2 ;

%Calculate diasmetic statistic
[R,P,T1u,Er,Ep] = diasmetic_rotations(T1o,T2o,F1,F2);

% Incorporate noise and apply the rotation
T1oe = ((X1n + E1) * V1);    % T1 with noise (before rotation)
T1r = T1oe * R;         % T1 after rotation
T2oe = (X2n + E2) * V2;        % T2 after noise (no rotation)

N = size(T1u,1);

Sobs  = (Er'*Er + Ep'*Ep) / (N);
Siobs = pinv(Sobs);
Lobs  = chol(Siobs,'lower');

Fd = norm(T1u*(P - R)*Lobs,'fro')^2;
Fp = zeros([1,n_perms]);

for ii = 1:n_perms
    perms = randperm(size(E1,1));
    Eperm = E1(perms,:);
    Xperm = X1n + Eperm;

    pD1 =  pinv(Z1);
    Bperm = pD1*Xperm;
    X1perm = Z1*Bperm;
    
    Tpm = X1perm * V1;
    [Tpu, ~, ~] = uniquetol(Tpm,1e-6, 'ByRows', true, 'PreserveRange', true);

    Fp(ii) = norm(Tpu*(P - R)*Lobs,'fro')^2;

end

%Studentization of the test statistic

Td  = (Fd - mean(Fp)) / std(Fp);
Tp  = (Fp - mean(Fp)) / std(Fp);

p = (sum(abs(Td) >= abs(Tp)) + 1) / (n_perms + 1);

end


function [R,P,T1u,Er,Ep] = diasmetic_rotations(T1o,T2o,F1,F2)

[T1u, ord1, ~] = uniquetol(T1o,1e-6, 'ByRows', true, 'PreserveRange', true);
[T2u, ord2, ~] = uniquetol(T2o,1e-6, 'ByRows', true, 'PreserveRange', true);

lvls1 = F1(ord1, 1);
lvls2 = F2(ord2, 1);

%Orient levels according to T1u - should it be opposite?
[~, perm_idx] = ismember(lvls1, lvls2);
n = numel(lvls1);
P = eye(n);
P = P(perm_idx, :);
T2ua = P * T2u;

M = T1u' * T2ua;
[Up, ~, Vp] = svd(M);
R = Up * Vp';  % Rotation matrix
Er = T1u * R - T2ua;

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

Ep = T1u*P - T2ua;

%D = diag(sign(diag(T1u'*T2u)));

%T1u = T1u*D;

%T1u = T1u ./vecnorm(T1u);

end