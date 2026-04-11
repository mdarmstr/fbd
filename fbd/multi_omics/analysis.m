% Add the MEDA toolbox to the path (ensure MEDA toolbox is installed)
addpath(genpath('../MEDA'));
load('ST002948.mat')
rng('shuffle');  
n_perms = 2;

F1 = blk1_mta(:,2);%(:,4);
F2 = blk2_mta(:,2);%(:,4);

% F1(:,[1,2]) = F1(:,[2,1]);
% F2(:,[1,2]) = F2(:,[2,1]);

X11 = trainblk11;
X11 = (X11 - mean(X11))./std(X11);

X22 = trainblk22;
X22 = (X22 - mean(X22))./std(X22);

X12 = testblk12;
X12 = (X12 - mean(X12))./std(X12);

X21 = testblk21;
X21 = (X21 - mean(X21))./std(X21);

% Factor out replicates
[T11, parglmo11] = parglm(X11, F1, 'Preprocessing', 0);% 'Model',[2,3]);
[T22, parglmo22] = parglm(X22, F2, 'Preprocessing', 0);%, 'Model',[2,3]);
 
[T12, parglmo12] = parglm(X12, F1, 'Preprocessing', 0);%, 'Model',[2,3]);
[T21, parglmo21] = parglm(X21, F2, 'Preprocessing', 0);%, 'Model',[2,3]);

% [T11, parglmo11] = parglm(X11, F1, 'Preprocessing', 0, 'Ordinal', [0,0,0,0],'Nested',[1,2]);
% [T22, parglmo22] = parglm(X22, F2, 'Preprocessing', 0, 'Ordinal', [0,1,0,0],'Model','interaction','Nested',[1,2]);
% 
% [T12, parglmo12] = parglm(X12, F1, 'Preprocessing', 0, 'Ordinal', [0,1,0,0],'Model','interaction','Nested',[1,2]);
% [T21, parglmo21] = parglm(X21, F2, 'Preprocessing', 0, 'Ordinal', [0,1,0,0],'Model','interaction','Nested',[1,2]);

mdl = fbd(parglmo11,parglmo22,n_perms);
mdl.opt()
mdl.pred_X1X2();
X12h = mdl.X1X2n;

X12 = zeros(size(X12h,1),size(X12h,2));

%Debug plot


% T1 = mdl.T1o*mdl.R;
% T2 = mdl.T2o;
% 
% gscatter(T1(:,1),T1(:,2),sort(ia1)); hold on;
% gscatter(T2(:,1),T2(:,2),sort(ia2)); hold off;

for ii = 1:size(F1,2)
    X12 = X12 + parglmo12.factors{ii}.matrix;
end

%X12 = uniquetol(X12,'ByRows',true);
%X12h = uniquetol(X12h,'ByRows',true);

[~, ia1] = uniquetol(X12, 1e-6, 'ByRows', true);
X12 = X12(sort(ia1), :);

[~, ia2] = uniquetol(X12h, 1e-6, 'ByRows', true);
X12h = X12h(sort(ia2),:);

recon = norm(X12 - X12h,'fro')^2 / norm(X12,'fro')^2;
recon2 = rv_coefficient(X12,X12h);

ax1 = subplot(3,1,1);
imagesc(X12)
colorbar
title('Nom')

ax2 = subplot(3,1,2);
imagesc(X12h)
colorbar
title('Rec')

ax3 = subplot(3,1,3);
imagesc(X12 - X12h)
colorbar
title('diff')
clims = [min([X12h(:); X12(:)]), max([X12h(:); X12(:)])];
set([ax1 ax2 ax3], 'CLim', clims)

disp(recon)
disp(recon2)
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