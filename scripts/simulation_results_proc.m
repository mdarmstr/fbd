%% SCRIPT: Repeated Positive & Negative Case Analysis
%  This script replicates the Positive and Negative case generation 
%  N times, collects the probability (p-value) for each run, and
%  plots the distribution of these p-values for each case.

% Ensure you have all the required functions in your path, including:
%  create_design, simuleMV, blockDiagonalSampling, parglm, fbd, etc.

clear; close all; clc;

%% Parameters
N = 100;           
reps = 10;
vars = 300;
levels = {[1, 2, 3,4,5,6],[1,2,3]};
rng('shuffle');

% Arrays to store p-values
pVals_Positive = zeros(N, 1);
pVals_Negative = zeros(N, 1);

%% Loop through N simulations
for i = 1:N
    
    % ------------------ POSITIVE CASE ------------------

    % Prepare random effect levels
    F = createDesign(levels,'Replicates',reps);

    X = zeros(size(F,1),vars);
    for ii = 1:length(levels{1})
        X(find(F(:,1) == levels{1}(ii)),:) = simuleMV(length(find(F(:,1) == levels{1}(ii))),vars,'LevelCorr',8) + repmat(randn(1,vars),length(find(F(:,1) == levels{1}(ii))),1);
    end

    % Shuffle row order, ensure approximately equal distribution of classes
    rp = randperm(size(X,1));
    X = X(rp,:);
    F = F(rp,:);

    % Split data into X1, X2
    [X1, X2] = blockDiagonalSampling(X, 0.4, 'both');
    F1 = F(1:size(X1,1), :);
    F2 = F(size(X1,1)+1:end, :);

    % Fit parglm
    [~, parglmo1] = parglm(X1, F1, 'Preprocessing', 1);
    [~, parglmo2] = parglm(X2, F2, 'Preprocessing', 1);

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
    reps = 4;
    vars = 120;

    F = createDesign(levels,'Replicates',reps);

    X = zeros(size(F,1),vars);
    for ii = 1:length(levels{1})
        X(find(F(:,1) == levels{1}(ii)),:) = simuleMV(length(find(F(:,1) == levels{1}(ii))),vars,'LevelCorr',8) + repmat(randn(1,vars),length(find(F(:,1) == levels{1}(ii))),1);
    end

    X1_neg = X;
    F1_neg = F;

    reps = 6;
    vars = 180;

    F = createDesign(levels, 'Replicates', reps);
    X = zeros(size(F,1),vars);
    for ii = 1:length(levels{1})
        X(find(F(:,1) == levels{1}(ii)),:) = simuleMV(length(find(F(:,1) == levels{1}(ii))),vars,'LevelCorr',8) + repmat(randn(1,vars),length(find(F(:,1) == levels{1}(ii))),1);
    end

    X2_neg = X;  % entire dataset for negative
    F2_neg = F;

    [~, parglmo1_neg] = parglm(X1_neg, F1_neg, 'Preprocessing', 1);
    [~, parglmo2_neg] = parglm(X2_neg, F2_neg, 'Preprocessing', 1);

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
% % Optionally, you can show a summary boxplot of the p-values:
% figure('Name','Boxplots of p-values','Color','w');
% boxplot([pVals_Positive, pVals_Negative], 'Labels', {'Positive','Negative'});
% ylabel('p-value');
% title('Boxplot of p-values across N simulations');

%% ------------------ Helper Functions ------------------

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
% PROTOTYPE PERMUTATION TEST CURRENTLY WORKING OKAY AS OF NOV 2025

X1 = parglmoA.data;
%X2 = parglmoB.data;

X1ne = parglmoA.residuals;
X2ne = parglmoB.residuals;

X1n = parglmoA.factors{fctrs(1)}.matrix;
X2n = parglmoB.factors{fctrs(1)}.matrix;

D1 = parglmoA.D(:,parglmoA.factors{fctrs(1)}.Dvars);
%D2 = parglmoB.D(:,parglmoB.factors{fctrs(1)}.Dvars);

% Scores calculation
[~, ~, V1] = svds(X1n, rank(X1n));
[~, ~, V2] = svds(X2n, rank(X2n));

% Data pre-treatment no longer appears to be necessary.
%[V1,T] = rotatefactors(V1,'Method','varimax','maxit',5000,'reltol',1e-6);
%V2 = rotatefactors(V2,'Method','varimax','maxit',5000,'reltol',1e-6);

T1o = X1n * V1;
T2o = X2n * V2;

%Calculate diasmetic statistic
[R,P,T1u,Er,Ep] = diasmetic_rotations(T1o,T2o,F1,F2);

%W = X1n*pinv(X1); %previous idea for permutation test.

Bhat = pinv(D1)*X1;
Xhat = D1*Bhat;
E    = X1 - Xhat;


% Incorporate noise and apply the rotation
T1oe = ((X1n + X1ne) * V1);    % T1 with noise (before rotation)
T1r = T1oe * R;         % T1 after rotation
T2oe = (X2n + X2ne) * V2;        % T2 after noise (no rotation)

N = size(Er,2);
Sobs  = (Er'*Er + Ep'*Ep) / (N * 2);

Siobs = pinv(Sobs);
Lobs  = chol(Siobs,'lower');


%(det(pinv((pinv(Er'*Er) + pinv(Ep'*Ep)))))^(N/2)*

%Fd = (det(2*pi*pinv(pinv(Er'*Er) + pinv(Ep'*Ep))))^(-1/2)*norm(T1u*(P - R)*Lobs,'fro')^2;
%Fd = norm(T1u*(P - R)*Lobs,'fro')^2; %/ norm(V1'*E'*E*V1,'fro')^2;

Fd = norm(((R*Lobs)'*(P*Lobs)),'fro')^2;

Fp = zeros([1,n_perms]);

for ii = 1:n_perms
    perms = randperm(size(E,1));
    Eperm = E(perms,:);
    %Xperm = X1(perms,:);
    Xperm = Xhat + Eperm;

    pD1 =  pinv(D1'*D1)*D1';
    Bperm = pD1*Xperm;
    X1perm = D1*Bperm;
    %Es = Xperm - X1perm;
    [~,~,Vpm] = svds(X1perm,rank(X1n));
        
    Tpm = (X1perm) * Vpm;
    [Rp,Pp,T1up,Erp,Epp] = diasmetic_rotations(Tpm,T2o,F1,F2);
    Sobs  = (Erp'*Erp + Epp'*Epp) / (N * 2);
    Siobs = pinv(Sobs);
    L  = chol(Siobs,'lower');

    Fp(ii) = norm(((Rp*L)'*(Pp*L)),'fro')^2;

end

p = (sum(Fp <= Fd) + 1) / (n_perms + 1);

%mu  = mean(Fp);
%sig = std(Fp);
%z_obs  = (Fd - mu) / sig;
%z_perm = (Fp - mu) / sig;
%p = mean(abs(z_perm) >= abs(z_obs));
%p = Fd;


% ascao1 = asca(parglmoA);
% ascao2 = asca(parglmoB);
% 
% scatter(ascao1.factors{1}.scores(:,1),ascao1.factors{1}.scores(:,2),[],F1(:,1))
% hold on;
% scatter(ascao2.factors{1}.scores(:,1),ascao2.factors{1}.scores(:,2),[],F2(:,1))
% hold off;

%scores(ascao1.factors{1}.scores,'ObsClass',F1(:,1),'Title','ASCA1')
%scores(ascao2.factors{1}.scores,'ObsClass',F2(:,1),'Title','ASCA2')

end

function F_KL = KL_divergence(Erp, Epp)
    n  = size(Erp,1);
    Sp = (Erp.'*Erp)/n;
    Sq = (Epp.'*Epp)/n;

    m   = size(Sp,1); %Regularization used here - not really important.
    lam = 1e-6*trace(Sp+Sq)/m;
    Sp  = Sp+lam*eye(m);
    Sq  = Sq+lam*eye(m);

    % Cholesky logdets - for probability calculations.
    Cp = chol(Sp,'lower'); Cq = chol(Sq,'lower');
    logdetSp = 2*sum(log(diag(Cp)));
    logdetSq = 2*sum(log(diag(Cq)));

    tr_SqinvSp = trace(Sq*pinv(Sp)); 
    F_KL = 0.5*(tr_SqinvSp-(logdetSp-logdetSq)-m);
end

function [R,P,T1u,Er,Ep] = diasmetic_rotations(T1o,T2o,F1,F2)

[T1u, ord1, ~] = uniquetol(T1o,1e-8, 'ByRows', true, 'PreserveRange', true);
[T2u, ord2, ~] = uniquetol(T2o,1e-8, 'ByRows', true, 'PreserveRange', true);

lvls1 = F1(ord1, 1);
lvls2 = F2(ord2, 1);

%Orient levels according to T1u - should it be opposite?
[~, perm_idx] = ismember(lvls1, lvls2);
n = numel(lvls1);
P = eye(n);
P = P(perm_idx, :);
T2u = P * T2u;

M = T1u' * T2u;
[Up, ~, Vp] = svd(M);
R = Up * Vp';  % Rotation matrix
Er = T1u * R - T2u;

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

Ep = T1u*P - T2u;

%T1u = T1u ./vecnorm(T1u);

end


