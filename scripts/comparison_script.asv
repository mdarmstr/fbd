%randn_gra(100,{[1,2,3]},1)
close all;
reps_pos = 150;
vars_pos = 100;
levels = {[1,2,3]};
nse = 1;
N = 1000; % Define the number of iterations for the main loop
err_pos = zeros(1,N);
err_neg = zeros(1,N);

for n = 1:N
    close all;

    reps   = reps_pos;
    vars   = vars_pos;

    F = createDesign(levels, 'Replicates', reps);
    X = zeros(size(F,1),vars);
    for ii = 1:length(levels{1})
        X(find(F(:,1) == levels{1}(ii)),:) = randn(length(find(F(:,1) == levels{1}(ii))),vars) + nse.*repmat(randn(1,vars),length(find(F(:,1) == levels{1}(ii))),1);
    end

    % Shuffle row order
    rp = randperm(size(X,1));
    X = X(rp,:);
    F = F(rp,:);

    % Split data into X1, X2
    [X1, X2] = blockDiagonalSampling(X, 0.4, 'both');
    F1 = F(1:size(X1,1), :);
    F2 = F(size(X1,1)+1:end, :);

    [F1s,idx1] = sort(F1);
    [F2s,idx2] = sort(F2);

    X1 = X1(idx1,:);
    X2 = X2(idx2,:);

    Z1 = ortho_code(F1s);
    Z2 = ortho_code(F2s);

    B1hat = pinv(Z1)*X1;
    X1n = Z1*B1hat;
    E1 = X1 - X1n;

    B2hat = pinv(Z2)*X2;
    X2n = Z2*B2hat;
    E2 = X2 - X2n;

    % Scores calculation
    [U1,S1,V1] = svds(X1n,rank(X1n));
    [U2,S2,V2] = svds(X2n,rank(X2n));

    T1 = U1*S1; T2 = U2*S2;

    M1 = U1'*Z1; M2 = U2'*Z2;

    [u,~,v] = svd(M1,"econ"); Dl1 = u*v';
    [u,~,v] = svd(M2,"econ"); Dl2 = u*v';

    T1d = T1*Dl1; T2d = T2*Dl2;

    [T1u, ~, ~] = uniquetol(T1d, 1e-4, 'ByRows', true, 'PreserveRange', true);
    [T2u, ~, ~] = uniquetol(T2d, 1e-4, 'ByRows', true, 'PreserveRange', true);

    [R,P,T1u,Er,Ep] = diasmetic_rotations(T1u,T2u,F1s,F2s);

    Sobs  = (Er'*Er + Ep'*Ep) / (N);
    Siobs = pinv(Sobs);
    Lobs  = chol(Siobs,'lower');

    err_pos(n) = norm(T1u*(P - R)*Lobs,'fro')^2;
    % figure(1)
    % hold on;
    % title('positive')
    % scatter(scrs1(:,1),scrs1(:,2),[],F1s,'filled');
    % scatter(scrs2(:,1),scrs2(:,2),[],F2s);
    % hold off;
    % disp('Positive')
    % disp(norm(Ep));

    %------------NEGATIVE CASE--------------%
    reps = floor(0.4*reps_pos);
    vars = floor(0.4*vars_pos);

    F = createDesign(levels,'Replicates',reps);

    X = zeros(size(F,1),vars);
    for ii = 1:length(levels{1})
        X(find(F(:,1) == levels{1}(ii)),:) = randn(length(find(F(:,1) == levels{1}(ii))),vars) + nse.*repmat(randn(1,vars),length(find(F(:,1) == levels{1}(ii))),1);
    end

    X1_neg = X;
    F1_neg = F;

    reps = floor(0.6*reps_pos);
    vars = floor(0.6*vars_pos);

    F = createDesign(levels, 'Replicates', reps);
    X = zeros(size(F,1),vars);
    for ii = 1:length(levels{1})
        X(find(F(:,1) == levels{1}(ii)),:) = randn(length(find(F(:,1) == levels{1}(ii))),vars) + nse.*repmat(randn(1,vars),length(find(F(:,1) == levels{1}(ii))),1);
    end

    X2_neg = X;  % entire dataset for negative
    F2_neg = F;

    [F1s,idx1] = sort(F1_neg);
    [F2s,idx2] = sort(F2_neg);

    X1 = X1_neg(idx1,:);
    X2 = X2_neg(idx2,:);

    Z1 = ortho_code(F1s);
    Z2 = ortho_code(F2s);

    B1hat = pinv(Z1)*X1;
    X1n = Z1*B1hat;
    E1 = X1 - X1n;

    B2hat = pinv(Z2)*X2;
    X2n = Z2*B2hat;
    E2 = X2 - X2n;

    [U1,S1,V1] = svds(X1n,rank(X1n));
    [U2,S2,V2] = svds(X2n,rank(X2n));

    T1 = U1*S1; T2 = U2*S2;

    M1 = U1'*Z1; M2 = U2'*Z2;

    [u,~,v] = svd(M1,"econ"); Dl1 = u*v';
    [u,~,v] = svd(M2,"econ"); Dl2 = u*v';

    T1d = T1*Dl1; T2d = T2*Dl2;

    [T1u, ~, ~] = uniquetol(T1d, 1e-4, 'ByRows', true, 'PreserveRange', true);
    [T2u, ~, ~] = uniquetol(T2d, 1e-4, 'ByRows', true, 'PreserveRange', true);

    [R,P,T1u,Er,Ep] = diasmetic_rotations(T1u,T2u,F1s,F2s);

    Sobs  = (Er'*Er + Ep'*Ep) / (N);
    Siobs = pinv(Sobs);
    Lobs  = chol(Siobs,'lower');

    err_neg(n) = norm(T1u*(P - R)*Lobs,'fro')^2;

    % scrs1 = (X1n+E1)*V1*P;
    % scrs2 = (X2n+E2)*V2;
    % figure(2);
    % hold on;
    % title('negative')
    % scatter(scrs1(:,1),scrs1(:,2),[],F1s,'filled');
    % scatter(scrs2(:,1),scrs2(:,2),[],F2s);
    % hold off;
    % disp('Negative')
    % disp(norm(Ep));

end

plot(log10(err_pos)); hold on; %blue should be lower
plot(log10(err_neg)); hold off;

[h,p,~,~] = ttest(err_neg,err_pos)

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

function [R,P,T1u,Er,Ep] = diasmetic_rotations(T1o,T2o,F1,F2)

% [T1u, ord1, ~] = uniquetol(T1o,1e-6, 'ByRows', true, 'PreserveRange', true);
% [T2u, ord2, ~] = uniquetol(T2o,1e-6, 'ByRows', true, 'PreserveRange', true);
% 
% lvls1 = F1(ord1, 1);
% lvls2 = F2(ord2, 1);
% 
% %Orient levels according to T1u - should it be opposite?
% [~, perm_idx] = ismember(lvls1, lvls2);
% n = numel(lvls1);
% P = eye(n);
% P = P(perm_idx, :);
% T2ua = P * T2u;

T1u = T1o;
T2u = T2o;

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

%D = diag(sign(diag(T1u'*T2u)));

%T1u = T1u*D;

%T1u = T1u ./vecnorm(T1u);

end

function [C, pairNames] = pairwise_contrasts(levels)
    levels = levels(:);
    cats = unique(levels);
    K = numel(cats);
    n = numel(levels);

    nContrasts = K*(K-1)/2;
    C = zeros(n, nContrasts);
    pairNames = strings(1, nContrasts);

    cIdx = 1;
    for i = 1:K
        for j = i+1:K
            a = cats(i);
            b = cats(j);

            col = zeros(n,1);

            ia = (levels == a);
            ib = (levels == b);

            na = sum(ia);
            nb = sum(ib);

            wa = 1/sqrt(na);
            wb = 1/sqrt(nb);

            col(ia) = +wa;
            col(ib) = -wb;

            % normalize to unit norm
            col = col / norm(col);

            C(:, cIdx) = col;
            pairNames(cIdx) = sprintf('%dvs%d', a, b);
            cIdx = cIdx + 1;
        end
    end
end

function Z2 = ortho_code(F2)

H2 = dummyvar(categorical(F2));
n_levels = sum(H2,1); %redundant - fix l8r
Z2 = H2 ./ sqrt(n_levels);

end