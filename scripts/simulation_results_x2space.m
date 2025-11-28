clear; close all; clc;

%% Parameters
N = 100;           

% Arrays to store p-values
pVals_Positive = zeros(N, 1);
pVals_Negative = zeros(N, 1);

reps = 10;
vars = 300;
levels = {[1, 2, 3,4,5],[1, 2, 3]};
rng("default");


%% Loop through N simulations
for i = 1:N
    %rng('shuffle');
    % ------------------ POSITIVE CASE ------------------

    % Prepare random effect levels
    F = createDesign(levels, 'Replicates', reps);
    
    X = zeros(size(F,1), vars);

    % Prepare random effect levels
    for ii = 1:length(levels{1})
        %X(find(F(:,1) == levels{1}(ii)),:) = simuleMV(length(find(F(:,1) == levels{1}(ii))),vars,'LevelCorr',8) + repmat(randn(1,vars),length(find(F(:,1) == levels{1}(ii))),1);
        X(find(F(:,1) == levels{1}(ii)),:) = randn(length(find(F(:,1) == levels{1}(ii))),vars) + repmat(randn(1,vars),length(find(F(:,1) == levels{1}(ii))),1);
    end
    % Shuffle row order
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
        %X(find(F(:,1) == levels{1}(ii)),:) = simuleMV(length(find(F(:,1) == levels{1}(ii))),vars,'LevelCorr',8) + repmat(randn(1,vars),length(find(F(:,1) == levels{1}(ii))),1);
        X(find(F(:,1) == levels{1}(ii)),:) = randn(length(find(F(:,1) == levels{1}(ii))),vars) + repmat(randn(1,vars),length(find(F(:,1) == levels{1}(ii))),1);
    end
    X1_neg = X;
    F1_neg = F;

    reps = 6;
    vars = 180;

    F = createDesign(levels, 'Replicates', reps);
    X = zeros(size(F,1),vars);
    for ii = 1:length(levels{1})
        %X(find(F(:,1) == levels{1}(ii)),:) = simuleMV(length(find(F(:,1) == levels{1}(ii))),vars,'LevelCorr',8) + repmat(randn(1,vars),length(find(F(:,1) == levels{1}(ii))),1);
        X(find(F(:,1) == levels{1}(ii)),:) = randn(length(find(F(:,1) == levels{1}(ii))),vars) + repmat(randn(1,vars),length(find(F(:,1) == levels{1}(ii))),1);
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

X1 = parglmoA.data;
%X2 = parglmoB.data;

X1ne = parglmoA.residuals;
X2ne = parglmoB.residuals;

X1n = parglmoA.factors{fctrs(1)}.matrix;
X2n = parglmoB.factors{fctrs(1)}.matrix;

D1 = parglmoA.D(:,parglmoA.factors{fctrs(1)}.Dvars);
%D2 = parglmoB.D(:,parglmoB.factors{fctrs(1)}.Dvars);



%[V1,T] = rotatefactors(V1,'Method','varimax','maxit',5000,'reltol',1e-12);
%V2 = rotatefactors(V2,'Method','varimax','maxit',5000,'reltol',1e-12);


[R,P,T1u,T2u,Er,Ep,V,T] = diasmetic_rotations(X1n, X2n, F1,F2,false);
N = size(T1u,1);

Sobs  = (Er'*Er + Ep'*Ep) / (N * 2);
Siobs = pinv(Sobs);
Lobs  = chol(Siobs,'lower');


[Q,~] = qr(T2u,'econ');

%J = Q*Q';

%[Qp,~] = qr(T1u*(P)*Lobs,'econ'); [Qr,~] = qr(T2u,'econ');


Fd = norm(T1u*(R - P)*Lobs,'fro')^2;

Fp = zeros([1,n_perms]);

Tpp = zeros(size(T1u,1),size(T1u,2),n_perms);

for ii = 1:n_perms
    perms = randperm(size(X1,1));
    Eperm = X1ne(perms,:);
    %Xperm = X1(perms,:);
    Xperm = X1n + Eperm;

    pD1 =  pinv(D1'*D1)*D1';
    Bperm = pD1*Xperm;
    X1perm = D1*Bperm;
    [~,~,V] = svds(X1perm,rank(X1n));
    Tpu = X1perm*V*T;
    
    [Rp,Pp,Tpu,T2u,Erp,Epp,~] = diasmetic_rotations(Tpu, X2n, F1, F2, true);
    Tpp(:,:,ii) = Tpu;

    S  = (Erp'*Erp + Epp'*Epp) / (N * 2);
    Si = pinv(S);
    L  = chol(Si,'lower');

    %[Qpp,~] = qr(Tpu*(Rp - eye(2))*L,'econ'); [Qrp,~] = qr(T2u,'econ');

    Fp(ii) = norm(Tpu*(Rp - Pp)*L,'fro')^2;
    
end

p = (sum(Fp >= Fd)+1) / (n_perms + 1); 
T1oe = []; T1r = []; T2oe = [];
%save_matrix_gif(Tpp,'animate.gif',0.25)


%p = Fd;

mu  = mean(Fp);
sig = std(Fp);
z_obs  = (Fd - mu) / sig;
z_perm = (Fp - mu) / sig;
p = mean(abs(z_perm) >= abs(z_obs));


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

function [R,P,T1u,T2u, Er,Ep,V1,T] = diasmetic_rotations(X1n,X2n,F1,F2,perm)

if perm == false
    %Calculate common subspace with respect to X_2
    [X1u, ord1, ~] = uniquetol(X1n, 1e-4, 'ByRows', true, 'PreserveRange', true);
    [X2u, ord2, ~] = uniquetol(X2n, 1e-4, 'ByRows', true, 'PreserveRange', true);

    [U,~,~] = svds(X2u,rank(X2n)); %COMMON SUBSPACE

    [V1,~,~] = svd(X1u'*U,'econ');
    [V3,T] = rotatefactors(V1,'Method','varimax','maxit',5000,'reltol',1e-12);
    %C = X1u'*U
    %V1 = U'*X1u;
    V2 = U'*X2u;

    %V1 = V1' ./vecnorm(V1');
    V2 = V2' ./vecnorm(V2');

    T1u = X1u*V3;
    T2u = X2u*V2;
else
    %Calculate error statistics based on scores
    [T1u, ord1, ~] = uniquetol(X1n, 1e-4, 'ByRows', true, 'PreserveRange', true);
    [X2u, ord2, ~] = uniquetol(X2n, 1e-4, 'ByRows', true, 'PreserveRange', true);
    [~,~,V2] = svds(X2u,rank(X2n));
    %V2 = U'*X2u;
    %V2 = V2' ./vecnorm(V2');
    T2u = X2u*V2;
    V1 = [];
end

lvls1 = F1(ord1, 1);
lvls2 = F2(ord2, 1);

[~, perm_idx] = ismember(lvls1, lvls2);
n = numel(lvls1);
P = eye(n);
P = P(perm_idx, :);
T2u = P * T2u;

% T1u = T1u ./vecnorm(T1u);
% T2u = T2u ./vecnorm(T2u);

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

end

function save_matrix_gif(P, filename, frameDelay)
%SCATTER_TENSOR_GIF  Animate scatter(P(:,1,k), P(:,2,k)) and save as GIF.
%
%   P is [nPoints x 2 x nFrames]
%   scatter_tensor_gif(P, "out.gif", 0.05)

    if nargin < 3
        frameDelay = 0.1;
    end

    [~, ~, nFrames] = size(P);

    % Precompute axis limits
    allX = P(:,1,:);
    allY = P(:,2,:);
    xlimVals = [min(allX(:)), max(allX(:))];
    ylimVals = [min(allY(:)), max(allY(:))];

    figure('Color','w');
    h = scatter(P(:,1,1), P(:,2,1), 36, 'filled');
    axis equal;
    xlim(xlimVals);
    ylim(ylimVals);
    axis off;

    drawnow;
    frame = getframe(gcf);
    [imind, cm] = rgb2ind(frame2im(frame), 256);
    imwrite(imind, cm, filename, 'gif', ...
        'Loopcount', inf, 'DelayTime', frameDelay);

    for k = 2:nFrames
        set(h, 'XData', P(:,1,k), ...
               'YData', P(:,2,k));

        drawnow;
        frame = getframe(gcf);
        [imind, cm] = rgb2ind(frame2im(frame), 256);
        imwrite(imind, cm, filename, 'gif', ...
            'WriteMode', 'append', 'DelayTime', frameDelay);
    end
end