%% FBD effect-size comparison

clear all; %#ok
load('bcdata.mat');

addpath(genpath('../MEDA'));
rng(1);
n_perms = 2; %placeholder for independence testing, not used for this
n_glm_perms = 1000;

F80 = D580(:,2);
F83 = D583(:,2);
Fa = [F80; F83];

[Z80,mu80,sd80] = preprocess2D(X580,'Preprocessing',2);
[Z83,mu83,sd83] = preprocess2D(X583,'Preprocessing',2);

[T80_fbd,parglmo80_fbd] = parglm(Z80,F80,'Preprocessing',0,'Permutations',n_glm_perms);
[T83_fbd,parglmo83_fbd] = parglm(Z83,F83,'Preprocessing',0,'Permutations',n_glm_perms);

mdl80to83 = fbd(parglmo80_fbd,parglmo83_fbd,n_perms);
mdl80to83.opt();
mdl80to83.pred_X1X2();

mdl83to80 = fbd(parglmo83_fbd,parglmo80_fbd,n_perms);
mdl83to80.opt();
mdl83to80.pred_X1X2();

results = table('Size',[0,17], ...
    'VariableTypes',[{'string'},repmat({'double'},1,16)], ...
    'VariableNames', ...
    {'Dataset','n','m','dff','dfe','SSf','SSe','SSt','MSf','MSe', ...
     'F','eta2','eps2','omg2','f','f_equal','p'});

% Column augmentation.
X580_col  = [Z80,   mdl80to83.X1X2n];
X580_cole = [Z80,   mdl80to83.X1X2e];
X583_col  = [mdl83to80.X1X2n,   Z83];
X583_cole = [mdl83to80.X1X2e,   Z83];

% Row augmentation: 580 samples first, then 583 samples.
X580_row  = [Z80;   mdl83to80.X1X2n];
X580_rowe = [Z80;   mdl83to80.X1X2e];
X583_row  = [mdl80to83.X1X2n;   Z83];
X583_rowe = [mdl80to83.X1X2e;   Z83];

% Full augmentation.
Xtn  = [Z80,mdl80to83.X1X2n;mdl83to80.X1X2n,Z83];
Xtne = [Z80,mdl80to83.X1X2e;mdl83to80.X1X2e,Z83];

datasets = {'X580','X583', ...
    'X580_col','X580_cole','X583_col','X583_cole', ...
    'X580_row','X580_rowe','X583_row','X583_rowe', ...
    'Xtn','Xtne'};

data = {Z80,Z83, ...
    X580_col,X580_cole,X583_col,X583_cole, ...
    X580_row,X580_rowe,X583_row,X583_rowe, ...
    Xtn,Xtne};

factors = {F80,F83, ...
    F80,F80,F83,F83, ...
    Fa,Fa,Fa,Fa, ...
    Fa,Fa};

for i = 1:numel(datasets)
    [n,m,dff,dfe,SSf,SSe,SSt,MSf,MSe,F,eta2,eps2,omg2,f,f_equal,p] = ...
        evaluate_glm(data{i},factors{i},0,n_glm_perms);

    results(i,:) = {datasets{i},n,m,dff,dfe,SSf,SSe,SSt,MSf,MSe, ...
        F,eta2,eps2,omg2,f,f_equal,p};
end

%% AI GENERATED .TEX TABLE WRITING FN -- CONFIRMED
fid = fopen('effect_results.tex','w');

% Open the LaTeX table: one text column, remaining columns numeric.
fprintf(fid,'%s\n', ...
    ['\begin{tabular}{l', repmat('r',1,width(results)-1), '}'], ...
    '\hline');

% Column headings
headers = strrep(results.Properties.VariableNames,'_','\_');
fprintf(fid,'%s\n', [strjoin(headers,' & '), ' \\'], '\hline');

% Data rows
for i = 1:height(results)
    label = strrep(char(results.Dataset(i)),'_','\_');
    numbers = compose('%.3g',results{i,2:end});
    line = [label, ' & ', strjoin(cellstr(numbers),' & '), ' \\'];
    fprintf(fid,'%s\n',line);
end

fprintf(fid,'%s\n','\hline','\end{tabular}');
fclose(fid);


function [n,m,dff,dfe,SSf,SSe,SSt,MSf,MSe,Fstat,eta2,eps2,omg2,f,f_equal,p] = evaluate_glm(X,factor,prep,permutations)
    
[T,parglmo] = parglm(X,factor, ...
        'Preprocessing',prep, ...
        'Permutations',permutations,...
        'Fmtc',3);

    [n,m] = size(parglmo.data);

    SSf = norm(parglmo.factors{1}.matrix,'fro')^2;
    SSe = norm(parglmo.residuals,'fro')^2;
    SSt = norm(parglmo.data,'fro')^2;

    source = string(T.Source);
    frow = source == "Factor 1";
    erow = source == "Residuals";

    dff = T.df(frow);
    dfe = T.df(erow);

    MSf = SSf/dff;
    MSe = SSe/dfe;

    Fstat = T.F(frow);
    p = parglmo.p(1);

    % Effect sizes.
    eta2 = SSf/SSt;
    eps2 = (SSf - dff*MSe)/(SSf + SSe);
    omg2 = (SSf - dff*MSe)/(SSt + MSe);
    f = sqrt(Fstat*dff/dfe);

    % Equally weighted separation of fitted factor-level centroids.
    [~,~,group] = unique(factor(:));
    K = max(group);

    A = parglmo.factors{1}.matrix;
    centroids = zeros(K,size(A,2));

    for k = 1:K
        centroids(k,:) = mean(A(group == k,:),1);
    end

    center_equal = mean(centroids,1);
    between_equal = mean(sum((centroids-center_equal).^2,2));

    f_equal = sqrt(between_equal/MSe);
end