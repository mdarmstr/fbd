function [cumpress,press,nze] = crossvalPls(x,y,varargin)

% Row-wise k-fold (rkf) cross-validation for square-prediction-errors computing in PLS.
%
% cumpress = crossvalpls(x,y) % minimum call
%
%
% See also: crossvalPlsDA, crossvalPca
%
%
% INPUTS:
%
% x: [NxM] billinear data set for model fitting
%
% y: [NxO] billinear data set of predicted variables
%
% Optional INPUTS:
%
% 'LVs': [1xA] Latent Variables considered (e.g. lvs = 1:2 selects the
%   first two LVs). By default, lvs = 0:rank(x)
%
% 'VarNumber': [1xK] Numbers of x-block variables selected. By default, VarNumber = M
%
% 'MaxBlock': [1x1] maximum number of blocks of samples (N by default)
%
% 'PreprocessingX': [1x1] preprocesing of the x-block
%       0: no preprocessing
%       1: mean centering
%       2: autoscaling (default)  
%
% 'PreprocessingY': [1x1] preprocesing of the y-block
%       0: no preprocessing
%       1: mean centering
%       2: autoscaling (default)  
%
% 'Plot': (bool) plot results
%       false: no plots.
%       true: plot (default)
%
% 'Selection': str
%   'Weights': filter method based on the PLS weights (W)
%   'AltWeights': filter method based on the PLS alternative weights (R)
%   'Regressors': filter method based on the PLS regression coefficients (beta)
%   'SR': filter method based on the selectivity ratio (by default)
%   'VIP': filter method based on Variance Importance in PLS Projection
%   'T2': wrapper method based on the Hotelling T2 statistic
%   'sPLS': embedded method based on sparse PLS
%
%
% OUTPUTS:
%
% cumpress: [AxK] Cumulative PRESS
%
% press: [AxKxO] PRESS per variable
%
% nze: [AxK] Non-zero elements in the regression coefficient matrix.
%
%
% EXAMPLE OF USE: Random data with structural relationship
% 
% X = simuleMV(20,10,'LevelCorr',8);
% Y = 0.1*randn(20,2) + X(:,1:2);
% keepXs = 1:10;
% [cumpress,press,nze] = crossvalPls(X,Y,'VarNumber',keepXs);
%
%
% coded by: Jose Camacho (josecamacho@ugr.es)
% last modification: 03/Feb/2025
%
% Copyright (C) 2025  University of Granada, Granada
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
M = size(x, 2);
O = size(y, 2);

% Introduce optional inputs as parameters (name-value pair) 
p = inputParser;
lat=0:rank(x);
addParameter(p,'LVs',lat'); 
addParameter(p,'VarNumber',M);
addParameter(p,'MaxBlock',N);
addParameter(p,'PreprocessingX',2);   
addParameter(p,'PreprocessingY',2);
addParameter(p,'Selection','SR'); 
addParameter(p,'Plot',true);   
parse(p,varargin{:});

% Extract inputs from inputParser for code legibility
lvs = p.Results.LVs;
keepXs = p.Results.VarNumber;
blocksr = p.Results.MaxBlock;
prepx = p.Results.PreprocessingX;
prepy = p.Results.PreprocessingY;
selection = p.Results.Selection;
opt = p.Results.Plot;

% Extract LVs and VarNumber length
A = length(lvs);
J =  length(keepXs);

% Convert column arrays to row arrays
if size(lvs,2) == 1, lvs = lvs'; end;
if size(keepXs,2) == 1, keepXs = keepXs'; end;

% Validate dimensions of input data
assert (isequal(size(y), [N O]), 'Dimension Error: parameter ''y'' must be N-by-O. Type ''help %s'' for more info.', routine(1).name);
assert (isequal(size(lvs), [1 A]), 'Dimension Error: parameter ''LVs'' must be 1-by-A. Type ''help %s'' for more info.', routine(1).name);
assert (isequal(size(keepXs), [1 J]), 'Dimension Error: parameter ''VarNumber'' must be 1-by-J. Type ''help %s'' for more info.', routine(1).name);
assert (isequal(size(blocksr), [1 1]), 'Dimension Error: parameter ''MaxBlock'' must be 1-by-1. Type ''help %s'' for more info.', routine(1).name);
assert (isequal(size(prepx), [1 1]), 'Dimension Error: parameter ''PreprocessingX'' must be 1-by-1. Type ''help %s'' for more info.', routine(1).name);
assert (isequal(size(prepy), [1 1]), 'Dimension Error: parameter ''PreprocessingY'' must be 1-by-1. Type ''help %s'' for more info.', routine(1).name);

% Preprocessing
lvs = unique(lvs);
keepXs = unique(keepXs);

% Validate values of input data
assert (isempty(find(lvs<0)), 'Value Error: parameter ''LVs'' must not contain negative values. Type ''help %s'' for more info.', routine(1).name);
assert (isequal(fix(lvs), lvs), 'Value Error: parameter ''LVs'' must contain integers. Type ''help %s'' for more info.', routine(1).name);
assert (isequal(fix(keepXs), keepXs), 'Value Error: parameter ''VarNumber'' must contain integers. Type ''help %s'' for more info.', routine(1).name);
assert (isequal(fix(blocksr), blocksr), 'Value Error: parameter ''MaxBlock'' must be an integer. Type ''help %s'' for more info.', routine(1).name);
assert (blocksr>2, 'Value Error: parameter ''MaxBlock'' must be above 2. Type ''help %s'' for more info.', routine(1).name);
assert (blocksr<=N, 'Value Error: parameter ''MaxBlock'' must be at most N. Type ''help %s'' for more info.', routine(1).name);


%% Main code

% Initialization
press = zeros(length(lvs),length(keepXs),O);
nze = zeros(length(lvs),length(keepXs));


rows = rand(1,N);
[a,rind]=sort(rows);
elemr=N/blocksr;

% Cross-validation
        
for i=1:blocksr
    
    indi = rind(round((i-1)*elemr+1):round(i*elemr)); % Sample selection
    i2 = ones(N,1);
    i2(indi)=0;
    sample = x(indi,:);
    calibr = x(find(i2),:); 
    sampley = y(indi,:);
    calibry = y(find(i2),:); 

    [ccs,av,st] = preprocess2D(calibr,'Preprocessing',prepx);
    [ccsy,avy,sty] = preprocess2D(calibry,'Preprocessing',prepy);
        
    scs = preprocess2Dapp(sample,av,'Scale',st);
    scsy = preprocess2Dapp(sampley,avy,'Scale',sty);
    
    if  ~isempty(find(lvs))
        
        for lv=1:length(lvs)

            for keepX=1:length(keepXs)
                
                if lvs(lv)
                    model = vpls(ccs,ccsy,'LVs',1:lvs(lv),'VarNumber',keepXs(keepX),'Selection',selection);

                    srec = scs*model.beta;
                    pem = scsy-srec;

                    press(lv,keepX,:) = squeeze(press(lv,keepX,:))' + sum(pem.^2,1);
					nze(lv,keepX) = nze(lv,keepX) + length(find(model.beta)); 
                else
                    press(lv,keepX,:) = squeeze(press(lv,keepX,:))' + sum(scsy.^2,1);
					nze(lv,keepX) = nze(lv,keepX) + M*O; 
                end
                
            end
            
        end
        
    else
        pem = scsy;
        press = press + ones(length(keepXs),1)*sum(pem.^2,1);
		nze = nze + ones(length(keepXs),1)*M*O;
    end
    
end

cumpress = sum(press,3);

%% Show results

if opt
    figh = plotVec(cumpress,'EleLabel',lvs,'XYLabel',{'#NZV','PRESS'},'PlotType','Lines','VecLabel',keepXs,'Color','jet'); 
end

