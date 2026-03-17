% Add the MEDA toolbox to the path (ensure MEDA toolbox is installed)
addpath(genpath('../MEDA'));
levels = {[1, 2, 3]};
vars_pos = 400;
reps_pos = 50;
nse = 1;
n_perms = 100;
rng('shuffle');  

recon = zeros(1,100);
for ii = 1:100
% F = createDesign(levels, 'Replicates', reps);
% X = zeros(size(F,1), vars);

% Prepare random effect levels
[X11, X22, X12, X21, F1, F2, szPos] = simul_data('pos', levels, reps_pos, vars_pos, nse, 0.4, 'both');

% Fit parglm
[~, parglmo1] = parglm(X11 - mean(X11), F1, 'Preprocessing', 0);
[~, parglmo2] = parglm(X22 - mean(X22), F2, 'Preprocessing', 0);

mdl = fbd(parglmo1,parglmo2,n_perms);
mdl.test()
mdl.pred_X1X2()

[~,parglmo12] = parglm(X12 - mean(X12),F1,'Preprocessing',0);

X = parglmo12.factors{1}.matrix;
%X = X12 - mean(X12);
Xh = mdl.X1X2n;

disp('hello')

%X = X12;
%Xh = mdl.X1X2e + mean(X22); % how to predict the mean, if we're comparing mean-centered data?

% X22 = parglmo2.factors{1}.matrix;
% 
% [~,~,Vh] = svds(Xh,4);
% [~,~,V2] = svds(X22,4);


%X = X./norm(X);
%Xh = Xh./norm(Xh);

%recon(ii) = rv_coefficient(X,Xh);
recon(ii) = norm(X - Xh,'fro')^2 / norm(X,'fro')^2;

if ii == 1
figure(1);

ax1 = subplot(3,1,1);
imagesc(X)
colorbar
title('Nom')

ax2 = subplot(3,1,2);
imagesc(Xh)
colorbar
title('Rec')

ax3 = subplot(3,1,3);
imagesc(X - Xh)
colorbar
title('diff')

clims = [min([Xh(:); X(:)]), max([Xh(:); X(:)])];
set([ax1 ax2 ax3], 'CLim', clims)

figure(2);
%PCA nominal;

[U,S,~] = svds(X,2);
T1 = U*S;

%PCA predicted;
hold on;
[U,S,~] = svds(Xh,2);
T2 = U*S;

gscatter(T1(:,1),T1(:,2),F1,[],'o');
gscatter(T2(:,1),T2(:,2),F1)


% 

end

end
disp(mean(recon))
disp(std(recon))

%mean_accuracy = (1-mean(recon))*100
%mean_stdev = std(recon)*100

%disp(p_Pos)
%disp('hello')
%% 
% 
% % % Calculate projections
% % % For dataset 1 (T1) and dataset 2 (T2) before rotation:
% % [U1,S1,V1] = svds(X1s, rank(X1s));
% % [U2,S2,V2] = svds(X2s, rank(X2s));
% % 
% % T1_orig = U1 * S1;  % T1 before rotation (from X1)
% % T2_orig = U2 * S2;  % T2 before rotation (from X2)
% % 
% % % Remove duplicate rows if desired
% % [T1u,ord1,~] = uniquetol(T1_orig, 'ByRows', true,'PreserveRange',true);
% % [T2u,ord2,~] = uniquetol(T2_orig, 'ByRows', true,'PreserveRange',true);
% % 
% % lvls1 = F1s(ord1,1);
% % lvls2 = F2s(ord2,1);
% % 
% % [~, perm_idx] = ismember(lvls1, lvls2);
% % % perm_idx = [2; 3; 1]  => means: row 1 of lvls1 is at row 2 of lvls2, etc.
% % 
% % % Step 2: Build permutation matrix
% % n = numel(lvls1);
% % P = eye(n);
% % P = P(perm_idx, :);   % permute the identity matrix
% % 
% % T2u = P*T2u;
% % 
% % % Compute the inner product between the unique rows and find the rotation
% % M = T1u' * T2u;
% % [U_tmp, ~, V_tmp] = svd(M);
% % R = U_tmp * V_tmp';  % Optimal rotation from Procrustes
% % 
% % % For "after rotation" we incorporate noise and then rotate T1
% % T1_noisy = ((X1s + X1es) * V1);  % T1 with noise (before rotation)
% % T1_rot = T1_noisy * R;          % T1 after applying the optimal rotation
% % 
% % T2_noisy = (X2s + X2es) * V2;      % T2 after adding noise (no rotation is applied)
% % 
% % % Extract the first two dimensions for plotting
% % T1_orig_data = T1_orig(:, 1:2);
% % T2_orig_data = T2_orig(:, 1:2);
% % 
% % T1_rot_data = T1_rot(:, 1:2);
% % T2_noisy_data = T2_noisy(:, 1:2);
% 
% %% === Prepare Class Labels and Colors ===
% 
% % Assuming F1 and F2 contain the class labels in their first column:
% ObsClass1 = F1(row_range1,1);
% ObsClass2 = F1(row_range2,1);
% 
% % Use the union of classes from both datasets to ensure consistency
% classes = unique([ObsClass1; ObsClass2]);
% colors = lines(length(classes));  % Use distinct colors
% 
% %% === Create Figure with Subplots ===
% 
% figure;
% 
% % --- Subplot 1: Before Rotation ---
% subplot(1,2,1);
% hold on;
% title('Before Rotation - Positive');
% xlabel('Component 1');
% ylabel('Component 2');
% 
% % Plot T1_orig (filled markers) by class
% for i = 1:length(classes)
%     idx = ObsClass1 == classes(i);
%     scatter(T1_noisy(idx,1), T1_noisy(idx,2), 50, colors(i,:), 'o', 'filled');
% end
% 
% % Plot T2_orig (hollow markers) by class
% for i = 1:length(classes)
%     idx = ObsClass2 == classes(i);
%     scatter(T2_noisy(idx,1), T2_noisy(idx,2), 50, colors(i,:), 'o');
% end
% 
% legend(arrayfun(@(c) ['Level ' num2str(c)], classes, 'UniformOutput', false), 'Location', 'best');
% hold off;
% 
% % --- Subplot 2: After Rotation ---
% subplot(1,2,2);
% hold on;
% title('After Rotation - Positive');
% xlabel('Component 1');
% ylabel('Component 2');
% 
% % Plot rotated T1 (filled markers) by class (using ObsClass1)
% for i = 1:length(classes)
%     idx = ObsClass1 == classes(i);
%     scatter(T1_rot_data(idx,1), T1_rot_data(idx,2), 50, colors(i,:), 'o', 'filled');
% end
% 
% % Plot noisy T2 (hollow markers) by class (using ObsClass2)
% for i = 1:length(classes)
%     idx = ObsClass2 == classes(i);
%     scatter(T2_noisy_data(idx,1), T2_noisy_data(idx,2), 50, colors(i,:), 'o');
% end
% 
% legend(arrayfun(@(c) ['Level ' num2str(c)], classes, 'UniformOutput', false), 'Location', 'best');
% hold off;
% 
% %% --- Formatting both subplots ---
% for ax = findall(gcf,'Type','axes')'
%     set(ax, 'FontName', 'Arial', ...
%             'FontSize', 14, ...
%             'LineWidth', 1, ...
%             'Box', 'off');
%     ax.XLabel.FontSize = 15;
%     ax.YLabel.FontSize = 15;
%     ax.Title.FontSize  = 15;
% end
% 
% % Legend formatting
% lgd = findall(gcf,'Type','Legend');
% set(lgd, 'FontSize', 12, 'Box', 'off');
% 
% % Save as cropped, vector PDF
% exportgraphics(gcf,'rotation_subplots.pdf','ContentType','vector');

%% Helper Functions

% Function to create plots for all datasets
function [X1, X2, Xoff12, Xoff21, F1, F2, sz] = simul_data(mode, levels, reps_pos, vars_pos, nse, splitFrac, splitMode)
%SIMUL_DATA  Simulate data for FBD experiments (positive/negative cases).
%
%   [X1,X2,F1,F2,sz,Xoff12,Xoff21] = simul_data('pos', levels, reps_pos, vars_pos, nse, 0.4, 'both');
%   [X1,X2,F1,F2,sz,Xoff12,Xoff21] = simul_data('neg', levels, reps_pos, vars_pos, nse, 0.4, 'both');

    if nargin < 7 || isempty(splitMode), splitMode = 'both'; end
    if nargin < 6 || isempty(splitFrac), splitFrac = 0.4; end

    mode = lower(string(mode));
    if mode == "positive", mode = "pos"; end
    if mode == "negative", mode = "neg"; end

    % Defaults for new outputs (keeps behavior consistent)
    Xoff12 = [];
    Xoff21 = [];

    switch mode
        case "pos"
            % ---- simulate one big matrix, shuffle, then blockDiagonalSampling ----
            [X, F] = local_sim_one(levels, reps_pos, vars_pos, nse);

            % Shuffle row order (exactly as in your code)
            rp = randperm(size(X,1));
            X = X(rp,:);
            F = F(rp,:);

            % Split data into X1, X2 and export off-diagonal blocks
            % NOTE: blockDiagonalSampling must support [b1,b2,off12,off21]
            [X1, X2, Xoff12, Xoff21] = smpl_blkdiag(X, splitFrac, splitMode);

            % Factor matrices sliced by the row split
            % (This matches your existing logic: first rows go with X1, remaining with X2)
            F1 = F(1:size(X1,1), :);
            F2 = F(size(X1,1)+1:end, :);

            sz = struct();
            sz.mode = 'pos';
            sz.X_full = size(X);
            sz.F_full = size(F);
            sz.X1 = size(X1);  sz.X2 = size(X2);
            sz.F1 = size(F1);  sz.F2 = size(F2);
            sz.Xoff12 = size(Xoff12);
            sz.Xoff21 = size(Xoff21);
            sz.splitFrac = splitFrac;
            sz.splitMode = char(splitMode);

        case "neg"
            % ---- simulate X1 and X2 independently (exactly like your code path) ----
            reps1 = floor(splitFrac * reps_pos);
            vars1 = floor(splitFrac * vars_pos);

            [X1, F1] = local_sim_one(levels, reps1, vars1, nse);

            reps2 = floor((1 - splitFrac) * reps_pos);
            vars2 = floor((1 - splitFrac) * vars_pos);

            [X2, F2] = local_sim_one(levels, reps2, vars2, nse);

            % Off-diagonal blocks are undefined in the independent (neg) simulation
            Xoff12 = [];
            Xoff21 = [];

            sz = struct();
            sz.mode = 'neg';
            sz.X1 = size(X1);  sz.X2 = size(X2);
            sz.F1 = size(F1);  sz.F2 = size(F2);
            sz.reps1 = reps1;  sz.vars1 = vars1;
            sz.reps2 = reps2;  sz.vars2 = vars2;
            sz.Xoff12 = [];
            sz.Xoff21 = [];
            sz.splitFrac = splitFrac;
            sz.splitMode = char(splitMode);

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

function rv = rv_coefficient(X, Y)
%RV_COEFFICIENT  Compute the RV coefficient between two data blocks
%
%   rv = rv_coefficient(X, Y)
%
% X : [n x p] data matrix
% Y : [n x q] data matrix
%
% Both matrices should be column-centered beforehand.

    % Cross-covariance
    XY = X' * Y;

    % RV coefficient
    rv = trace(XY * XY') / sqrt( trace((X' * X)^2) * trace((Y' * Y)^2) );
end