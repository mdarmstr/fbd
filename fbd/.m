clear all;
addpath(genpath('../MEDA'));
load('ST002948.mat');
rng('shuffle');
n_perms = 2;
rng_state = rng; % Use the same initial random state for both FBD fits.

n1 = size(trainblk11,1);
p1 = size(trainblk11,2);
F1 = blk1_mta(:,2);
F2 = blk2_mta(:,2);
Fa = [F1; F2];

methods = {'Original: blockwise GLM'; 'Pooled GLM before splitting'};
recon = zeros(2,1);
recon2 = zeros(2,1);

for method = 1:2
    if method == 1
        % Original methodology: standardize and fit each block separately.
        X11 = (trainblk11 - mean(trainblk11))./std(trainblk11);
        X22 = (trainblk22 - mean(trainblk22))./std(trainblk22);
        X12 = (testblk12 - mean(testblk12))./std(testblk12);
        X21 = (testblk21 - mean(testblk21))./std(testblk21);
    else
        % Current methodology: standardize and fit the complete matrix first.
        Xt = [trainblk11, testblk12;
              testblk21, trainblk22];
        Xt = (Xt - mean(Xt))./std(Xt);
        [Tt, parglmot] = parglm(Xt, Fa, 'Preprocessing', 0);
        Xtr = parglmot.factors{1}.matrix;

        X11 = Xtr(1:n1,1:p1);
        X12 = Xtr(1:n1,p1+1:end);
        X21 = Xtr(n1+1:end,1:p1);
        X22 = Xtr(n1+1:end,p1+1:end);
    end

    [T11, parglmo11] = parglm(X11, F1, 'Preprocessing', 0);
    [T22, parglmo22] = parglm(X22, F2, 'Preprocessing', 0);
    [T12, parglmo12] = parglm(X12, F1, 'Preprocessing', 0);
    [T21, parglmo21] = parglm(X21, F2, 'Preprocessing', 0);

    rng(rng_state);
    mdl = fbd(parglmo11, parglmo22, n_perms);
    mdl.opt();
    mdl.pred_X1X2();
    X12h = mdl.X1X2n;

    if method == 1
        % Original target: sum of the separately fitted factor matrices.
        Xtarget = zeros(size(X12));
        for ii = 1:size(F1,2)
            Xtarget = Xtarget + parglmo12.factors{ii}.matrix;
        end
    else
        % Preserve the target in the supplied pooled-GLM script: X12r.
        Xtarget = X12;
        % A global factor block need not be centered within this row subset.
        % If FBD predicts locally centered effects, its matching target would
        % instead be parglmo12.factors{1}.matrix. Do not silently switch targets.
    end

    % Keep corresponding sample rows in both methods; no independent uniquetol.
    assert(isequal(size(Xtarget),size(X12h)), ...
        'Target and prediction sizes differ. Check FBD output row mapping.');
    assert(all(isfinite(Xtarget(:))) && all(isfinite(X12h(:))), ...
        'Nonfinite values: check inputs and zero standard deviations.');
    assert(norm(Xtarget,'fro') > 0, 'Target has zero norm.');
    recon(method) = norm(Xtarget-X12h,'fro')^2 / norm(Xtarget,'fro')^2;
    recon2(method) = rv_coefficient(Xtarget,X12h);

    figure('Name',methods{method});
    ax1 = subplot(3,1,1);
    imagesc(Xtarget); colorbar; title('Nom');
    ax2 = subplot(3,1,2);
    imagesc(X12h); colorbar; title('Rec');
    ax3 = subplot(3,1,3);
    imagesc(Xtarget-X12h); colorbar; title('diff');
    clims = [min([Xtarget(:);X12h(:)]), max([Xtarget(:);X12h(:)])];
    if clims(1) == clims(2), clims = clims + [-1 1]; end
    set([ax1 ax2], 'CLim', clims);
    dlim = max(abs(Xtarget(:)-X12h(:)));
    if dlim == 0, dlim = 1; end
    set(ax3, 'CLim', [-dlim dlim]);
end

results = table(methods, recon, recon2, ...
    'VariableNames', {'Method','RelativeSquaredError','RV'});
disp(results);

% Export a LaTeX table; include with \input{fbd_comparison.tex}.
fid = fopen('fbd_comparison.tex','w');
assert(fid ~= -1, 'Could not open fbd_comparison.tex for writing.');
fprintf(fid, '%s\n', '\begin{table}[ht]');
fprintf(fid, '%s\n', '\centering');
fprintf(fid, '%s\n', ['\caption{FBD reconstruction of the off-diagonal block. ' ...
    'Each method is evaluated against its own factor target, using all matched ' ...
    'sample rows. The pooled method uses the complete matrix during preprocessing ' ...
    'and is a diagnostic experiment. RV is calculated after column centering.}']);
fprintf(fid, '%s\n', '\label{tab:fbd-comparison}');
fprintf(fid, '%s\n', '\begin{tabular}{lrr}');
fprintf(fid, '%s\n', '\hline');
fprintf(fid, '%s\n', 'Method & Relative squared error & RV \\');
fprintf(fid, '%s\n', '\hline');
for method = 1:2
    fprintf(fid, '%s & %.6f & %.6f %s\n', ...
        methods{method}, recon(method), recon2(method), '\\');
end
fprintf(fid, '%s\n', '\hline');
fprintf(fid, '%s\n', '\end{tabular}');
fprintf(fid, '%s\n', '\end{table}');
fclose(fid);
disp('Wrote fbd_comparison.tex');

function rv = rv_coefficient(X,Y)
    % Center explicitly: a subset of the pooled factor matrix may not be centered.
    X = X - mean(X,1);
    Y = Y - mean(Y,1);
    XY = X'*Y;
    denominator = sqrt(trace((X'*X)^2)*trace((Y'*Y)^2));
    if denominator == 0
        rv = NaN;
    else
        rv = trace(XY*XY') / denominator;
    end
end