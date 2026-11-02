function [X11,X22,X12,F1,F2] = simulate_blocks(n_levels,reps,vars,nse,splitFrac)
F = createDesign({1:n_levels},'Replicates',reps);
X = zeros(size(F,1),vars);

group_means = simuleMV(n_levels,vars,'LevelCorr',10);

for level = 1:n_levels
    rows = find(F(:,1)==level);
    X(rows,:) = nse*repmat(group_means(level,:),numel(rows),1) ...
        + randn(numel(rows),vars);
end

order = randperm(size(X,1));
X = X(order,:);
F = F(order,:);

[X11,X22,X12,~] = smpl_blkdiag(X,splitFrac,'both');

F1 = F(1:size(X11,1),:);
F2 = F(size(X11,1)+1:end,:);
end