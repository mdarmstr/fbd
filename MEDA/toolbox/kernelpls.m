function model = kernelpls(XX,XY,varargin)

% Kernel algorithm for Partial Least Squares. References:
% F. Lindgren, P. Geladi and S. Wold, J. Chemometrics, 7, 45 (1993).
% S. De Jong and C.J.F. Ter Braak, J. Chemometrics, 8, 169 (1994).
% B.S. Dayal and J.F. MacGregor. J. Chemometrics, 11, 73�85 (1997). Main
% code is almost copy-pasted from the last reference.
%
% model = kernelpls(XX,XY)     % minimum call
%
% See also: simpls, pcaEig, asca, gpls, sparsepls2
%
%
% INPUTS:
%
% XX: [MxM] cross-product X'*X
%
% XY: [MxO] cross-product X'*Y
%
%
% Optional INPUTS (parameter):
%
% 'LVs': [1xA] Latent Variables considered (e.g. lvs = 1:2 selects the
%   first two LVs). By default, lvs = 0:size(XX)
%
%
% OUTPUTS:
%
% model: structure that contains model information
%   var: [1x1] xcs sum of squares
%   lvs: [1xA] latent variable numbers
%   loads: [MxA] matrix of x-loadings P
%   yloads: [OxA] matrix of y-loadings Q
%   weights: [MxA] matrix of weights W
%   altweights: [MxA] matrix of alternative weights R
%   beta: [MxO] matrix of regressors
%   type: 'PLS'
%
%
% EXAMPLE OF USE: Random data with structural relationship
%
% X = simuleMV(20,10,'LevelCorr',8);
% Y = 0.1*randn(20,2) + X(:,1:2);
% Xcs = preprocess2D(X,'Preprocessing',2);
% Ycs = preprocess2D(Y,'Preprocessing',2);
% lvs = 1:10;
% model = kernelpls(Xcs'*Xcs,Xcs'*Ycs,'LVs',lvs);
%
%
% coded by: Jose Camacho (josecamacho@ugr.es)
% last modification: 18/Nov/2024
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
M = size(XX, 1);
O = size(XY, 2);

% Introduce optional inputs as parameters (name-value pair) 
p = inputParser;
LVS = 0:size(XX,1);
addParameter(p,'LVs',LVS);  
parse(p,varargin{:});

% Extract inputs from inputParser for code legibility
lvs = p.Results.LVs;

% Convert column arrays to row arrays
if size(lvs,2) == 1, lvs = lvs'; end;

% Preprocessing
lvs = unique(lvs);
lvs(find(lvs==0)) = [];
lvs(find(lvs>size(XX,1))) = [];
A = length(lvs);

% Validate dimensions of input data
assert (isequal(size(XX), [M M]), 'Dimension Error: parameter ''XX'' must be M-by-M. Type ''help %s'' for more info.', routine(1).name);
assert (isequal(size(XY), [M O]), 'Dimension Error: parameter ''YY'' must be M-by-O. Type ''help %s'' for more info.', routine(1).name);
assert (isequal(size(lvs), [1 A]), 'Dimension Error: parameter ''LVs'' must be 1-by-A. Type ''help %s'' for more info.', routine(1).name);

% Validate values of input data
assert (isempty(find(lvs<0)) && isequal(fix(lvs), lvs), 'Value Error: parameter ''LVs'' must contain positive integers. Type ''help %s'' for more info.', routine(1).name);


%% Main code

W=[];
P=[];
Q=[];
R=[];
for i=1:max(lvs), % A=number of PLS components to be computed
    if O==1 % if there is a single response variable, compute the
        w=XY; % X-weights as shown here
    else % else
        [C,D]=eig(XY'*XY); % ?rst compute the eigenvectors of YTXXTX
        q=C(:,find(diag(D)==max(diag(D)))); %find the eigenvector corresponding to the largest eigenvalue
        w=(XY*q); % compute X-weights
    end
    w=w/sqrt(w'*w); % normalize w to unity
    r=w; % loop to compute ri
    for j=1:i-1
        r=r-(P(:,j)'*w)*R(:,j);
    end
    tt=(r'*XX*r); % compute tTt
    p=(r'*XX)'/tt; % X-loadings
    q=(r'*XY)'/tt; % Y-loadings
    XY=XY-(p*q')*tt; % XTY de?ation
    W=[W w]; % storing loadings and weights
    P=[P p];
    Q=[Q q];
    R=[R r];
end

% Postprocessing
W = W(:,lvs);
P = P(:,lvs);
Q = Q(:,lvs);
R = R(:,lvs);
beta = R*Q';

model.var = trace(XX);
model.lvs = 1:size(P,2);
model.loads = P;
model.yloads = Q;
model.weights = W;
model.altweights = R;
model.beta = beta;
model.type = 'PLS';
