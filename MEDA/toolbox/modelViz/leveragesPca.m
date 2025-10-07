
function [L,E] = leveragesPca(x,varargin)

% Compute and plot the leverages of variables in the PCA model
%
% L = leveragesPca(x) % minimum call
%
%
% INPUTS:
%
% x: [NxM] billinear data set for model fitting
%
%
% Optional INPUTS:
%
% 'Pcs': [1xA] Principal Components considered (e.g. pcs = 1:2 selects the
%   first two PCs). By default, pcs = 1:rank(xcs)
%
% 'Preprocessing': [1x1] preprocesing of the data
%       0: no preprocessing
%       1: mean centering
%       2: autoscaling (default) 
%
% 'VarsLabel': [Mx1] name of the variables (numbers are used by default)
%
% 'VarsClass: [Mx1] groups for different visualization (a single group 
%   by default)
%
%
% OUTPUTS:
%
% L: [Mx1] leverages of the variables
%
% E: [Mx1] residuals of the variables
%
%
% EXAMPLE OF USE: Random leverages
%
% A = cell(1, 10);
% 
% for i = 1:10
%     A{i} = ['A_{', num2str(i), '}'];
% end
% 
% X = simuleMV(20,10,'LevelCorr',8);
% L = leveragesPca(X,'PCs',1,'VarsLabel',A);
%
%
% coded by: Jose Camacho (josecamacho@ugr.es)
% last modification: 15/Jan/2025
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

%% Parameters checking

% Set default values
routine=dbstack;
assert (nargin >= 1, 'Error in the number of arguments. Type ''help %s'' for more info.', routine(1).name);
N = size(x, 1);
M = size(x, 2);

% Introduce optional inputs as parameters (name-value pair) 
p = inputParser;
PCS = 1:rank(x);
addParameter(p,'PCs',PCS);  
addParameter(p,'Preprocessing',2);
Label = [1:M];
addParameter(p,'VarsLabel',Label);
Classes = ones(M,1);
addParameter(p,'VarsClass',Classes);     
parse(p,varargin{:});

% Extract inputs from inputParser for code legibility
pcs = p.Results.PCs;
prep = p.Results.Preprocessing;
label = p.Results.VarsLabel;
classes = p.Results.VarsClass;

% Convert row arrays to column arrays
if size(label,1) == 1,     label = label'; end;
if size(classes,1) == 1, classes = classes'; end;

% Convert column arrays to row arrays
if size(pcs,2) == 1, pcs = pcs'; end;

% Preprocessing
pcs = unique(pcs);
pcs(find(pcs==0)) = [];
A = length(pcs);

% Validate dimensions of input data
assert (A>0, 'Dimension Error: parameter ''Pcs'' with non valid content. Type ''help %s'' for more info.', routine(1).name);
assert (isequal(size(pcs), [1 A]), 'Dimension Error: parameter ''Pcs'' must be 1-by-A. Type ''help %s'' for more info.', routine(1).name);
assert (isequal(size(prep), [1 1]), 'Dimension Error: parameter ''preprocessing'' must be 1-by-1. Type ''help %s'' for more info.', routine(1).name);
assert (isequal(size(label), [M 1]), 'Dimension Error: parameter ''varsLabel'' must be M-by-1. Type ''help %s'' for more info.', routine(1).name); 
assert (isequal(size(classes), [M 1]), 'Dimension Error: parameter ''VarsClass'' must be M-by-1. Type ''help %s'' for more info.', routine(1).name); 
  
% Validate values of input data
assert (isempty(find(pcs<0)) && isequal(fix(pcs), pcs), 'Value Error: parameter ''Pcs'' must contain positive integers. Type ''help %s'' for more info.', routine(1).name);


%% Main code

xcs = preprocess2D(x,'Preprocessing',prep);
model = pcaEig(xcs,'PCs',pcs);
P = model.loads;
T = model.scores;
        
%L = diag(P*P');
%E = sum((xcs-T*P').^2);
L = sum((T*P').^2)./sum(xcs.^2);

%% Show results

plotVec(L, 'EleLabel',label, 'ObsClass',classes, 'XYLabel',{'Variables','Leverages'});
        