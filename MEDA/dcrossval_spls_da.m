function [AUCm,AUC,lvso,keepXso] = dcrossval_spls_da(x,y,varargin)

% Row-wise k-fold (rkf) double cross-validation in SPLS-DA, restricted to 
% one response categorical variable of two levels. Reference:
% J. Camacho, J. Gonz�lez-Mart�nez and E. Saccenti.
% Rethinking cross-validation in SPLS. Submitted to Journal of Chemometrics. 
% The algorithm uses repetitions of the dCV loop to estimate the stability: 
% see Szymanska, E., Saccenti, E., Smilde, A.K., Weterhuis, J. Metabolomics 
% (2012) 8: 3. It also corrects the classification limit following Richard 
% G. Brereton, J. Chemometrics 2014; 28: 213�225
%
% AUCm = dcrossval_spls_da(x,y) % minimum call
% [AUCm,AUC,lvso,keepXso] = 
% dcrossval_spls_da(x,y,'LatVars',lvs,'KeepXBlock',keepXs,'Alpha',alpha,'MaxBlock',blocks_r,'PreprocessingX',prepx,'PreprocessingY',prepy,'Repetition',rep,'Option',opt) % complete call
%
%
% INPUTS:
%
% x: [NxM] billinear data set for model fitting
%
% y: [Nx1] billinear data set of one categorical variable with two levels.
%
% Optional INPUTS (parameters):
%
% 'LatVars': [1xA] Latent Variables considered (e.g. lvs = 1:2 selects the
%   first two LVs). By default, lvs = 0:rank(x)
%
% 'KeepXBlock': [1xK] Numbers of x-block variables kept per latent variable modeled. By default,
% 	keepXs = 1:M
%
% 'Alpha': [1x1] Trade-off controlling parameter that goes from -1 (maximum 
%   completeness), through 0 (pure prediction, by default) to 1 (maximum 
%   parsimony) 
%
% 'MaxBlock': [1x1] maximum number of blocks of samples (the minimum number
%   of observations of a class by default)
%
% 'PreprocesingX': [1x1] preprocesing of the x-block
%       0: no preprocessing
%       1: mean centering
%       2: autoscaling (default)  
%
% 'PreprocesingY': [1x1] preprocesing of the y-block
%       0: no preprocessing
%       1: mean centering
%       2: autoscaling (default)  
%
% 'Repetition': [1x1] number of repetitios for stability
%
% 'Option': [1x1] options for data plotting
%       0: no plots
%       1: bar plot (default)
%
%
% OUTPUTS:
%
% AUCm: [1x1] Mean Area Under the ROC 
%
% AUC: [rep x 1] Area Under the ROC
%
% lvso: [rep x blocks_r] optimum number of LVs in the inner loop
%
% keepXso: [rep x blocks_r] optimum number of keepXs in the inner loop
%
%
% EXAMPLE OF USE: Random data with structural relationship
%
% X = simuleMV(20,10,'LevelCorr',8);
% X = [X 0.1*randn(20,10) + X];
% Y = 2*(0.1*randn(20,1) + X(:,1)>0)-1;
% lvs = 0:10;
% keepXs = 1:10;
% [AUCm,AUC,lvso,keepX] = dcrossval_spls_da(X,Y,'LatVars',lvs,'KeepXBlock',keepXs,'MaxBlock',5)
% [AUCm_simple,AUC_simple,lvso_simple,keepX_simple] = dcrossval_spls_da(X,Y,'LatVars',lvs,'KeepXBlock',keepXs,'Alpha',0.5,'MaxBlock',5)
% [AUCm_complete,AUC_complete,lvso_complete,keepX_complete] = dcrossval_spls_da(X,Y,'LatVars',lvs,'KeepXBlock',keepXs,'Alpha',-0.5,'MaxBlock',5)
%
%
% coded by: Jose Camacho Paez (josecamacho@ugr.es)
% last modification: 22/Apr/24
%
% Copyright (C) 2024  University of Granada, Granada
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.


%% Arguments checking

% Set default values
routine=dbstack;
assert (nargin >= 2, 'Error in the number of arguments. Type ''help %s'' for more info.', routine(1).name);
N = size(x, 1);
O = size(y, 2);
M = size(x, 2);

 vals = unique(y);
 rep2 = sort(histc(y,vals),'descend');
 N2 = rep2(2);

% Introduce optional inputs as parameters (name-value pair) 
p = inputParser;
lat=0:rank(x);
addParameter(p,'LatVars',lat'); 
keep = 1:M;
addParameter(p,'KeepXBlock',keep);
addParameter(p,'Alpha',0);
addParameter(p,'MaxBlock', max(3,round(N2/2)));
addParameter(p,'PreprocessingX',2);   
addParameter(p,'PreprocessingY',2);
addParameter(p,'Repetition',10);
addParameter(p,'Option',1);   
parse(p,varargin{:});

% Extract inputs from inputParser for code legibility

lvs = p.Results.LatVars;
alpha = p.Results.Alpha;
keepXs = p.Results.KeepXBlock;
blocks_r = p.Results.MaxBlock;
prepx = p.Results.PreprocessingX;
prepy = p.Results.PreprocessingY;
rep = p.Results.Repetition;
opt = p.Results.Option;

% Extract LatVars and Gamma length
A = length(lvs);
J =  length(keepXs);

% Convert column arrays to row arrays
if size(lvs,2) == 1, lvs = lvs'; end;
if size(keepXs,2) == 1, keepXs = keepXs'; end;


% Validate dimensions of input data
assert (isequal(size(y), [N 1]), 'Dimension Error: parameter ''y'' must be N-by-1. Type ''help %s'' for more info.', routine(1).name);
assert (isequal(size(lvs), [1 A]), 'Dimension Error: parameter ''LatVars'' must be 1-by-A. Type ''help %s'' for more info.', routine(1).name);
assert (isequal(size(keepXs), [1 J]), 'Dimension Error: parameter ''KeepXBlock'' must be 1-by-J. Type ''help %s'' for more info.', routine(1).name);
assert (isequal(size(alpha), [1 1]), 'Dimension Error: parameter ''Alpha'' must be 1-by-1. Type ''help %s'' for more info.', routine(1).name);
assert (isequal(size(blocks_r), [1 1]), 'Dimension Error: parameter ''MaxBlock'' must be 1-by-1. Type ''help %s'' for more info.', routine(1).name);
assert (isequal(size(prepx), [1 1]), 'Dimension Error: parameter ''PreprocessingX'' must be 1-by-1. Type ''help %s'' for more info.', routine(1).name);
assert (isequal(size(prepy), [1 1]), 'Dimension Error: parameter ''PreprocessingY'' must be 1-by-1. Type ''help %s'' for more info.', routine(1).name);
assert (isequal(size(rep), [1 1]), 'Dimension Error: parameter ''Repetition'' must be 1-by-1. Type ''help %s'' for more info.', routine(1).name);
assert (isequal(size(opt), [1 1]), 'Dimension Error: parameter ''Option'' must be 1-by-1. Type ''help %s'' for more info.', routine(1).name);

% Preprocessing
lvs = unique(lvs);
keepXs = unique(keepXs);

% Validate values of input data

assert (isempty(find(y~=1 & y~=-1)), 'Value Error: parameter ''y'' must not contain values different to 1 or -1. Type ''help %s'' for more info.', routine(1).name);
assert (isempty(find(lvs<0)), 'Value Error: parameter ''LatVars'' must not contain negative values. Type ''help %s'' for more info.', routine(1).name);
assert (isequal(fix(lvs), lvs), 'Value Error: parameter ''LatVars'' must contain integers. Type ''help %s'' for more info.', routine(1).name);
assert (isequal(fix(keepXs), keepXs), 'Value Error: parameter ''KeepXBlock'' must contain integers. Type ''help %s'' for more info.', routine(1).name);
assert (alpha>=-1 & alpha<=1, 'Value Error: parameter ''Alpha'' must contain values in [-1, 1]. Type ''help %s'' for more info.', routine(1).name);
assert (isequal(size(blocks_r), [1 1]), 'Dimension Error: parameter ''MaxBlock'' must be 1-by-1. Type ''help %s'' for more info.', routine(1).name);
assert (isequal(fix(blocks_r), blocks_r), 'Value Error: parameter ''MaxBlock'' must be an integer. Type ''help %s'' for more info.', routine(1).name);
assert (blocks_r>3, 'Value Error: parameter ''MaxBlock'' must be above 3. Type ''help %s'' for more info.', routine(1).name);
assert (blocks_r<=N, 'Value Error: parameter ''MaxBlock'' must be at most N. Type ''help %s'' for more info.', routine(1).name);


%% Main code

for j=1:rep
    % Cross-validation
    
    y1 = find(y==1);
    yn1 = find(y==-1);
    
    rows = rand(1,length(y1));
    [a,r_ind1]=sort(rows);
    elem_r1=length(y1)/blocks_r;
    
    rows = rand(1,length(yn1));
    [a,r_indn1]=sort(rows);
    elem_rn1=length(yn1)/blocks_r;
    
    % Cross-validation
    
    for i=1:blocks_r
        
        ind_i1 = r_ind1(round((i-1)*elem_r1+1):round(i*elem_r1)); % Sample selection
        i2 = ones(length(y1),1);
        i2(ind_i1)=0;
        val = x(y1(ind_i1),:);
        rest = x(y1(find(i2)),:);
        val_y = y(y1(ind_i1),:);
        rest_y = y(y1(find(i2)),:);
        
        ind_in1 = r_indn1(round((i-1)*elem_rn1+1):round(i*elem_rn1)); % Sample selection
        i2 = ones(length(yn1),1);
        i2(ind_in1)=0;
        val = [val;x(yn1(ind_in1),:)];
        rest = [rest;x(yn1(find(i2)),:)];
        val_y = [val_y;y(yn1(ind_in1),:)];
        rest_y = [rest_y;y(yn1(find(i2)),:)];
        
        [ccs,av,st] = preprocess2D(rest,'Preprocessing',prepx);
        %[ccs_y,av_y,st_y] = preprocess2D(rest_y,prepy);
        ccs_y = rest_y;
        
        [kk,m1] = preprocess2D(ccs(find(rest_y==1),:),'Preprocessing',1);  % additional subtraction of class mean
        [kk,mn1] = preprocess2D(ccs(find(rest_y==-1),:),'Preprocessing',1);
        ccs = preprocess2Dapp(ccs,(m1+mn1)/2);
        
        vcs = preprocess2Dapp(val,av,'SDivideTest',st);
        vcs = preprocess2Dapp(vcs,(m1+mn1)/2);
        
        %vcs_y = preprocess2Dapp(val_y,av_y,st_y);
        vcs_y = val_y;
        
        [AUCt,nze] =  crossval_spls_da(rest,rest_y,'LatVars',lvs,'KeepXBlock',keepXs,'MaxBlock',blocks_r-1,'PreprocessingX',prepx,'PreprocessingY',prepy,'Option',0);
        
        cumpressb = (abs(alpha)-1)*AUCt/max(max(AUCt)) + alpha*nze/max(max(nze));
        
        [l,k]=find(cumpressb==min(min(cumpressb)));
        lvso(j,i) = lvs(l(1));
        keepXso(j,i) = keepXs(k(1));
        
        if lvso(j,i)~=0
            
            model = sparsepls2(ccs, ccs_y, lvso(j,i), keepXso(j,i)*ones(size(1:lvso(j,i))), O*ones(size(1:lvso(j,i))), 500, 1e-10, 1, 0);
            beta = model.R*model.Q';
            
            sr = vcs*beta;
            srec1(ind_i1') = sr(1:length(ind_i1));
            srecn1(ind_in1') = sr(length(ind_i1)+1:end);
            
        else
            keepXso(j,i) = nan;
            srec1(ind_i1') = 0;
            srecn1(ind_in1') = 0;
        end
        
    end
    
    [~,~,~,AUC(j)] = perfcurve(y([y1;yn1]),[srec1';srecn1'],1);
end

AUCm = mean(AUC);

%% Show results

if opt == 1
    fig_h = plot_vec(AUC,'XYLabel',{'#Repetition','AUC'},'Option',11);
end


